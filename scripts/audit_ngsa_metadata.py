"""Extract explicit source lineage; do not infer measurement units from names."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'data/pilot_nk/ngsa/Sheet_229_Udegi'
out=ROOT/'outputs/udegi/reconnaissance'
ns={'g':'http://www.geosoft.com/schema/geo'}
rows=[]
for path in sorted(folder.rglob('*.grd.xml')):
    root=ET.parse(path).getroot()
    dataset=root.find('.//g:dataset',ns)
    parameters=[dict(p.attrib) for p in dataset.findall('.//g:parameter',ns)]
    processes=[{k:p.get(k) for k in ['name','displaystring','date']} for p in dataset.findall('.//g:process',ns)]
    sources=sorted({p['value'] for p in parameters if p.get('name')=='GRIDMASK::GRDIN'})
    # Projection units describe horizontal coordinates, not measured quantities.
    unit_nodes=[dict(e.attrib) for e in dataset.iter() if e.tag.split('}')[-1].lower() in ['units','unit','zunits','valueunits','dataunits']]
    row=dict(file=path.name,metadata=str(path),sources=sources,processes=processes,
             unit_nodes=unit_nodes,measurement_units_status='Not established by this sidecar audit',
             magnetic_processing_note='Source filename includes GradEnhanced; enhancement algorithm and physical units not specified here' if 'TMI' in path.name else None)
    rows.append(row)
    print(path.name, 'source:', '; '.join(sources), '| measurement units not confirmed')
report=out/'ngsa_metadata_audit.json'
report.write_text(json.dumps(rows,indent=2))
lines=['# NGSA Udegi metadata findings','',
       'Horizontal coordinates are WGS 84 / UTM zone 32N, in metres. These metres are not geophysical measurement units.','',
       'Sidecars document reprojection and polygon masking. The recorded processing dates are not acquisition dates.','',
       'The magnetic input filename includes GradEnhanced. Do not assume unprocessed total-field data, a particular derivative, or a particular enhancement algorithm.','',
       'The sidecars inspected do not establish potassium %, equivalent thorium/uranium ppm, or magnetic nT. Keep values labelled as source values until product documentation or NGSA confirms units and scaling.','',
       '## Source grids','']
for r in rows:
    lines.append(f'- {r["file"]}: '+ '; '.join(r['sources']))
lines += ['', '## Request to data provider','',
          'Please confirm units and scaling for Sheet 229 TMI, potassium, equivalent thorium and equivalent uranium; the magnetic gradient-enhancement method; acquisition dates, flight-line spacing and terrain clearance; radiometric calibration/corrections; and permitted reuse.']
(out/'ngsa_metadata_findings.md').write_text('\n'.join(lines),encoding='utf-8')
print(report)
