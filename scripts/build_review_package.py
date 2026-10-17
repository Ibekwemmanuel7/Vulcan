"""Consolidate existing exploratory evidence; never create occurrence labels."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/udegi/reconnaissance'
cells=json.loads((out/'spatial_review_cells.geojson').read_text())['features']
native=json.loads((out/'native_cell_comparison.json').read_text())
terrain=json.loads((out/'cell_terrain.json').read_text())
features=[]
lines=['# Udegi R21–R22 geological review package','',
       'Status: unverified spatial patterns. These areas are not occurrences, training labels, mineral predictions or ranked exploration targets.','',
       'Selected after visual inspection of potassium. R20 and R23 are neighbouring comparisons, not verified barren controls.','',
       '| Cell | Centre longitude | Centre latitude | Native K median | Native eTh median | Native eU median | Elevation median (m) |',
       '|---|---:|---:|---:|---:|---:|---:|']
for label in ['R20','R21','R22','R23']:
    feature=next(f for f in cells if f['properties']['review_id']==label)
    ring=feature['geometry']['coordinates'][0][:-1]
    lon=sum(p[0] for p in ring)/len(ring)
    lat=sum(p[1] for p in ring)/len(ring)
    vals={r['layer']:r['median'] for r in native['results'] if r['cell']==label}
    t=next(r for r in terrain['cells'] if r['cell']==label)
    lines.append(f'| {label} | {lon:.6f} | {lat:.6f} | {vals["POTASSIUM"]:.3f} | {vals["eTHORIUM"]:.3f} | {vals["eURANIUM"]:.3f} | {t["elevation_m_q25_median_q75"][1]:.1f} |')
    if label in ['R21','R22']:
        feature['properties'].update(status='unverified_geological_review_area',
            centre_longitude=lon,centre_latitude=lat,
            coordinate_note='Geometric cell centre, not a sampling location or mineral occurrence',
            native_medians=vals,terrain=t)
        features.append(feature)
lines += ['', '## Evidence and limitations','',
    '- R21 and R22 have higher native potassium and equivalent thorium medians than both comparison cells. Uranium contrast is smaller.',
    '- The pattern persists without resampling geophysical values to 10 m.',
    '- Geophysical comparisons use vegetation-filtered subsets; terrain summaries use entire cells. They have different spatial support.',
    '- Cell elevations descend toward the river; median slopes are similar. This does not distinguish bedrock from transported material.',
    '- The regional geological PDF is 1:1,000,000 with questionable embedded datum metadata. Its overlay is approximate context only.',
    '- Metadata audit confirms reprojection and masking, but not measured-value units. Magnetic source filename includes GradEnhanced; its enhancement algorithm remains unspecified. See ngsa_metadata_findings.md.',
    '- Spatially correlated grid cells are not independent observations. No statistical significance or mineral accuracy has been established.',
    '', '## Next evidence needed','',
    '1. Detailed geological mapping or qualified geological review of these boundaries: rock units, contacts, regolith and river deposits.',
    '2. Independently documented occurrence or sample coordinates with mineral descriptions, uncertainty, source and permitted use.',
    '3. If field inspection is arranged, record surface material and distinguish in-place outcrop from transported sediment; laboratory results are needed for commodity confirmation.',
    '', '## Files','',
    '- review_areas_R21_R22.geojson: WGS84 polygon boundaries, geometric centres and descriptive measurements.',
    '- native_cell_comparison.json: native-resolution measurements and sampling method.',
    '- cell_terrain.json: terrain statistics and preview path.',
    '', '## Expansion','',
    'Minnesota, USA, and other mining regions remain planned after this pilot. Select commodity and boundary and verify local datasets before transfer evaluation.']
(out/'review_areas_R21_R22.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=features),indent=2))
report=out/'R21_R22_review.md'
report.write_text('\n'.join(lines),encoding='utf-8')
print(report)
print(out/'review_areas_R21_R22.geojson')
