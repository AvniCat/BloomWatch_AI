"""Assemble and chunk the BloomWatch knowledge base.

Sources:
  1. CMFRI-documented bloom episodes extracted from Annual Reports (dataset_1.csv)
  2. Static guidance authored here (mitigation, forecast interpretation, calibration)
  3. Model performance context

Chunking strategy: one document per semantic unit (one bloom episode, one guide),
each with metadata so retrieval can filter by region / topic.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import APP_ROOT

# ------------- STATIC KNOWLEDGE (curated) -------------

STATIC_DOCS: list[dict] = [
    {
        "id": "how_to_interpret_forecast",
        "topic": "forecast_interpretation",
        "text": (
            "BloomWatch AI produces a probability between 0 and 1 that a harmful "
            "algal bloom will occur in the next 8-day window. The number is a raw "
            "XGBoost output. On the 2024 hold-out year the model achieved AUC 0.83, "
            "Brier score 0.09, and Expected Calibration Error 0.07 — meaning when "
            "it predicts 40 percent, blooms actually happen close to 40 percent of "
            "the time. We tested three post-hoc calibration methods including a "
            "novel outcome-anchored one using CMFRI closure records; none improved "
            "on the raw probability. Farmers can trust the forecast number directly."
        ),
    },
    {
        "id": "risk_bands",
        "topic": "risk_bands",
        "text": (
            "Risk bands used in the BloomWatch app: Low = P(bloom next week) below "
            "0.15. Elevated = 0.15 to 0.35. High = 0.35 to 0.55. Very High = above "
            "0.55. Elevated or higher means the farmer should consider bringing "
            "forward the harvest or delaying seeding by a week."
        ),
    },
    {
        "id": "mitigation_general",
        "topic": "mitigation",
        "text": (
            "Standard shellfish farm mitigation steps when a bloom is forecast: "
            "(1) Bring forward any planned harvest by 3-5 days if animals are near "
            "market weight. (2) Move floating rafts to deeper water where possible; "
            "blooms concentrate in the top 2-3 metres. (3) Reduce feeding rates to "
            "avoid additional nutrient loading. (4) Monitor visual water clarity "
            "and colour (brownish-red = Trichodesmium, greenish = Chlorella or "
            "Chattonella). (5) Coordinate with local CMFRI extension centre for "
            "confirmation before initiating emergency harvest."
        ),
    },
    {
        "id": "mitigation_shellfish_specific",
        "topic": "mitigation",
        "text": (
            "Shellfish (green mussel, oyster, clam) accumulate biotoxins from "
            "dinoflagellate blooms. Rule of thumb: after a confirmed bloom, wait "
            "at least 14 days after visual water clearance before resuming harvest "
            "for human consumption. Blood clam and green mussel are most vulnerable "
            "to paralytic shellfish poisoning (PSP) toxins from Alexandrium spp. "
            "and Pyrodinium bahamense. Trichodesmium blooms mainly cause hypoxia "
            "and are less toxic but still cause mass mortality via oxygen depletion "
            "at bloom collapse (typically 3-7 days after peak bloom)."
        ),
    },
    {
        "id": "why_forecast_is_uncertain",
        "topic": "confidence",
        "text": (
            "Sources of forecast uncertainty: (a) MODIS Aqua satellite retired in "
            "Feb 2025; current chlorophyll comes from Suomi-NPP VIIRS which has "
            "slightly different sensor characteristics. (b) IMD rainfall data "
            "comes from a scraped district-level dashboard; the first weekly "
            "refresh reports cumulative-since-monsoon-onset totals, subsequent "
            "runs report the true 8-day delta. (c) The training dataset covers "
            "only 5 years (2020-2024) with 460 weekly rows, so sample size for "
            "any given season is small. (d) CMFRI documented only 4 harvest "
            "closure events in that period — the forecast target is chlorophyll-a "
            "threshold OR CMFRI-documented event, dominated by the chlorophyll "
            "signal."
        ),
    },
    {
        "id": "data_sources",
        "topic": "provenance",
        "text": (
            "BloomWatch AI uses four public datasets: (1) NASA Suomi-NPP VIIRS "
            "Level-3 mapped 8-day chlorophyll-a (formerly MODIS-Aqua until Feb "
            "2025). (2) NASA VIIRS Level-3 mapped 8-day sea surface temperature. "
            "(3) IMD district-level cumulative rainfall from mausam.imd.gov.in. "
            "(4) CMFRI Annual Reports 2016-2024 for confirmed bloom events. All "
            "data is free and public; the pipeline refreshes every Friday via "
            "GitHub Actions."
        ),
    },
    {
        "id": "coverage",
        "topic": "coverage",
        "text": (
            "Current coverage: Kerala and Karnataka coasts of India. Kerala is "
            "defined as latitude 8.0-12.8N, longitude 74.5-77.5E. Karnataka is "
            "12.8-15.0N, 73.5-75.5E. Coastal districts included: Kerala — "
            "Thiruvananthapuram, Kollam, Alappuzha, Ernakulam, Thrissur, "
            "Malappuram, Kozhikode, Kannur, Kasargod, Kottayam. Karnataka — "
            "Dakshin Kannada, Udupi, Uttar Kannada. Notable aquaculture "
            "estuaries within coverage include Ashtamudi Lake in Kollam "
            "(Ramsar site, major clam and green mussel farming area), "
            "Vembanad Lake in Kottayam-Ernakulam-Alappuzha, and the "
            "Nethravathi-Gurupura estuary near Mangaluru. Model does not "
            "currently cover Tamil Nadu, Andhra Pradesh, Goa, or Odisha."
        ),
    },
    {
        "id": "shellfish_symptoms",
        "topic": "diagnosis",
        "text": (
            "Signs that a bloom is affecting your shellfish stock, in order of "
            "how quickly they appear:\n"
            "(1) MUSSELS — check for gaping: healthy green mussels stay tightly "
            "closed when handled. If shells hang open and don't close when "
            "tapped, the animal is stressed or dying. Weak byssal threads "
            "(the beard-like fibres that anchor mussels to the rope or "
            "substrate) that pull away easily is another early sign.\n"
            "(2) OYSTERS — reduced filtration rate: healthy oysters push water "
            "visibly through the shell gape. Slow or stopped pumping indicates "
            "distress. A brownish or greyish tint on the mantle, instead of "
            "pale cream, suggests toxin uptake.\n"
            "(3) CLAMS — buried clams that fail to withdraw the siphon when "
            "disturbed are dying. Clams found on the sediment surface (not "
            "buried) are almost always dead or dying.\n"
            "(4) ALL SPECIES — sudden mass mortality (more than 5% of stock in "
            "24 hours) is the clearest sign. Smell — rotten or sulfurous odor "
            "indicates hypoxia-driven death, often 3-7 days after a "
            "Trichodesmium bloom peaks. Colour of the flesh — normal is pale, "
            "greenish-brown flesh suggests toxin exposure.\n"
            "If you observe any of these, contact your local CMFRI extension "
            "centre immediately and STOP harvesting for human consumption."
        ),
    },
    {
        "id": "species_field_guide",
        "topic": "diagnosis",
        "text": (
            "Bloom species identification by visual cues (field guide for "
            "farmers, not a lab test):\n"
            "TRICHODESMIUM (also called sea sawdust): water turns brownish-red "
            "or rust-coloured. Small clumps look like fine sawdust or reddish "
            "streaks floating at the surface. Slight sulphur or hay-like "
            "smell. Common along Kerala coast in March-May. Not directly "
            "toxic to humans but causes fish and shellfish mortality via "
            "oxygen depletion when the bloom collapses (usually 3-7 days "
            "after peak).\n"
            "NOCTILUCA SCINTILLANS (sea sparkle): water appears greenish or "
            "reddish in daylight and glows blue-green when disturbed at "
            "night. Slimy or gelatinous texture. Causes hypoxia at high "
            "concentrations and increases ammonia, damaging shellfish.\n"
            "CHATTONELLA: water turns yellow-green to golden brown. "
            "Fish gills and mussel gill filaments become visibly damaged; "
            "look for fish with red or pink gills. Toxic to fish and can "
            "cause acute mortality.\n"
            "ALEXANDRIUM / PYRODINIUM: red-brown discoloration. These are "
            "the dinoflagellates that cause Paralytic Shellfish Poisoning "
            "(PSP) — the shellfish look normal but accumulate toxins that "
            "can be fatal to humans who eat them. If PSP is suspected, "
            "shellfish MUST be lab-tested before sale — do not rely on "
            "visual inspection.\n"
            "IN ALL CASES — do not rely on visual identification alone for "
            "food safety. Send a water sample and shellfish sample to "
            "your local CMFRI or state fisheries laboratory for "
            "confirmation before making harvest decisions."
        ),
    },
    {
        "id": "emergency_contacts",
        "topic": "contacts",
        "text": (
            "Emergency contacts for bloom or shellfish mortality events:\n"
            "KERALA — CMFRI Headquarters, Kochi: 0484-2394867, "
            "extension@cmfri.org.in. Kerala Department of Fisheries "
            "Vigilance & Support helpline: 1077 (state disaster helpline) "
            "or 0471-2308973 (Directorate of Fisheries). Kerala Marine "
            "Products Testing Lab: contact through Kollam or Ernakulam "
            "district fisheries offices.\n"
            "KARNATAKA — CMFRI Mangaluru Research Centre: 0824-2424152. "
            "Karnataka Department of Fisheries Directorate (Mangaluru): "
            "0824-2426003. Coastal district fisheries offices: Dakshin "
            "Kannada (0824-2426003), Udupi (0820-2534077), Uttar "
            "Kannada / Karwar (08382-226225).\n"
            "GOA — outside BloomWatch coverage but for reference: National "
            "Institute of Oceanography (NIO), Dona Paula: 0832-2450264. "
            "Goa Department of Fisheries: 0832-2437245.\n"
            "TAMIL NADU — CMFRI Chennai / Mandapam: 04573-241433.\n"
            "NATIONAL — Ministry of Earth Sciences INCOIS (Indian "
            "National Centre for Ocean Information Services): "
            "1800-425-1245, isg@incois.gov.in — they issue coastal "
            "hazard advisories including HAB alerts.\n"
            "If shellfish poisoning is suspected in humans (numbness, "
            "tingling, difficulty breathing) — call 108 (national medical "
            "emergency) immediately and take the patient plus a sample of "
            "the suspect shellfish to the nearest hospital."
        ),
    },
    {
        "id": "safe_disposal",
        "topic": "mitigation",
        "text": (
            "Safe disposal and cleanup after a bloom-related mortality event:\n"
            "(1) DO NOT eat or sell any of the affected stock, even if some "
            "individuals look healthy — biotoxins can be present without "
            "obvious symptoms.\n"
            "(2) SEPARATE clearly-dead animals from potentially-recoverable "
            "ones. Move the survivors to clean, well-oxygenated water if "
            "possible (deeper areas, or areas with tidal flushing).\n"
            "(3) DISPOSE of dead shellfish by deep burial (at least 1 metre "
            "below ground) at least 50 metres from any water body. Do NOT "
            "compost, do NOT feed to animals, do NOT return to the water. "
            "Cover the burial pit with lime (calcium oxide) if available "
            "to neutralise biological activity.\n"
            "(4) CLEAN farm equipment — ropes, floats, cages, and boats "
            "that contacted bloom water with a 1% bleach solution or hot "
            "water (60°C+). Rinse thoroughly and sun-dry for at least "
            "48 hours before re-use.\n"
            "(5) REPORT the event to your local CMFRI extension centre "
            "with (a) date and time first noticed, (b) estimated "
            "mortality percentage, (c) water color and any smell, "
            "(d) species affected. This helps CMFRI track outbreaks and "
            "improve future forecasts.\n"
            "(6) DOCUMENT for insurance or government relief: photograph "
            "the mortality, keep a written log of losses, and get a "
            "written confirmation from the CMFRI officer that visits."
        ),
    },
    {
        "id": "model_limitations",
        "topic": "limitations",
        "text": (
            "HONEST LIMITATIONS of the BloomWatch AI model — read this before "
            "answering any question about what the model can or cannot do.\n"
            "\n"
            "(1) TRICHODESMIUM BLIND SPOT — CRITICAL: BloomWatch CANNOT reliably "
            "detect Trichodesmium blooms from satellite chlorophyll-a alone. "
            "Trichodesmium is a nitrogen-fixing cyanobacterium whose photopigments "
            "(phycoerythrin, phycocyanin) sit OUTSIDE the standard chl-a retrieval "
            "bands used by VIIRS-SNPP. Cross-validation against 4 CMFRI-documented "
            "events (2020-2024) shows that only 1 of 4 events crossed our chl-a > 2 "
            "mg/m3 threshold. The 2023 Kochi and 2024 Kochi Trichodesmium blooms "
            "were BOTH missed by the satellite proxy (chl-a mean 0.19 and 0.13 "
            "respectively — well below threshold). Trichodesmium blooms are "
            "exactly the blooms most damaging to Kerala's shellfish because they "
            "cause hypoxia-driven mass mortality at bloom collapse. Farmers "
            "relying on BloomWatch alone would receive FALSE-NEGATIVE warnings "
            "for these events. Always cross-check with visual water color "
            "(brownish-red, rust, sawdust-like clumps) and CMFRI advisories.\n"
            "\n"
            "(2) BLOOM DEFINITION IS A PROXY: The model predicts elevated-"
            "chlorophyll weeks (chl-a > 2 mg/m3 per 8-day satellite window), NOT "
            "confirmed toxic HAB events. The paper title should be read as "
            "'elevated bloom risk forecasting,' not 'toxic HAB prediction.'\n"
            "\n"
            "(3) SMALL DATASET: Only 5 years of data (2020-2024, 460 weekly rows "
            "across 2 regions). Cannot claim generalisation across ENSO cycles or "
            "multi-decadal monsoon trends. Only 4 CMFRI-documented bloom events "
            "in the window.\n"
            "\n"
            "(4) BIDIRECTIONAL CALIBRATION ERROR: The model UNDER-predicts moderate "
            "risk (predictions near 0.15 correspond to empirical rate of 0.28) and "
            "OVER-predicts high risk (predictions near 0.65 correspond to empirical "
            "rate of 0.43). This is why we present risk BANDS rather than raw "
            "percentages, and why any raw probability above 0.15 should be treated "
            "as an elevated-caution flag.\n"
            "\n"
            "(5) CLOUD COVER: Optical satellite retrievals fail under persistent "
            "monsoon cloud cover (June-September). The 2022 Karnataka event was "
            "missed for this reason.\n"
            "\n"
            "(6) COVERAGE: Kerala and Karnataka coasts only. No forecast is "
            "available for Tamil Nadu, Andhra Pradesh, Goa, Gujarat, or any "
            "other Indian state at this time.\n"
            "\n"
            "When a user asks whether the model can detect a SPECIFIC bloom type, "
            "SPECIES, or REGION — check this document first. Do not claim "
            "detection capability that the model does not have. Being honest about "
            "the Trichodesmium blind spot is more valuable to the farmer than "
            "false reassurance."
        ),
    },
]


# ------------- HISTORICAL EVENTS FROM CMFRI -------------

def _load_cmfri_events() -> list[dict]:
    """Extract dated bloom events from dataset_1.csv."""
    candidates = [
        Path("/Users/avnisingh/ai essentials/hab-prediction_offline_data/dataset_1.csv"),
    ]
    csv = next((p for p in candidates if p.exists()), None)
    if csv is None:
        return []

    df = pd.read_csv(csv)
    # Keep only rows explicitly mentioning bloom / mortality / kill in notes
    bloom_rows = df[df["dominant_taxa_or_notes"].str.contains(
        r"bloom|mortalit|kill|Trichodesmium|Noctiluca|Diatoma|Gymnodinium|Gonyaulax|Alexandrium",
        case=False, na=False,
    )]
    docs = []
    seen = set()
    for _, r in bloom_rows.iterrows():
        key = (str(r.get("data_year")), str(r.get("region", ""))[:30])
        if key in seen: continue
        seen.add(key)
        note = str(r.get("dominant_taxa_or_notes", "")).strip()
        loc = str(r.get("region", "")).strip() or "unknown location"
        year = str(r.get("data_year", "unknown"))
        report = str(r.get("source_report", "CMFRI Annual Report"))
        docs.append({
            "id": f"cmfri_event_{report}_{year}_{loc[:20]}".replace(" ", "_"),
            "topic": "historical_event",
            "text": (
                f"CMFRI documented a harmful algal bloom event in {year} at {loc}. "
                f"Details: {note}. (Source: {report})"
            ),
            "region": ("Kerala" if any(k in loc for k in ["Kerala","Kochi","Cochin","Alappuzha"])
                       else "Karnataka" if any(k in loc for k in ["Karnataka","Mangaluru","Mangalore","Udupi","Karwar"])
                       else "other"),
        })
    return docs


# ------------- PUBLIC API -------------

def build_corpus() -> list[dict]:
    """Return the full document set for indexing."""
    corpus = list(STATIC_DOCS)
    corpus.extend(_load_cmfri_events())
    return corpus


def chunk_text(text: str, target_chars: int = 800) -> list[str]:
    """Simple sentence-boundary chunking. Most docs already fit in one chunk."""
    if len(text) <= target_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 > target_chars and buf:
            chunks.append(buf.strip()); buf = ""
        buf += " " + s
    if buf: chunks.append(buf.strip())
    return chunks


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Total docs: {len(corpus)}")
    by_topic = {}
    for d in corpus:
        by_topic.setdefault(d.get("topic", "?"), 0)
        by_topic[d["topic"]] += 1
    for t, n in sorted(by_topic.items()):
        print(f"  {t:20s} {n}")
    print(f"\nSample doc:")
    print(f"  id: {corpus[0]['id']}")
    print(f"  topic: {corpus[0]['topic']}")
    print(f"  text: {corpus[0]['text'][:200]}…")
