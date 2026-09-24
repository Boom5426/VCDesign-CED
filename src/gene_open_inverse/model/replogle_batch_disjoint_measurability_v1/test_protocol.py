import json
from pathlib import Path
def main():
 c=json.loads((Path(__file__).parent/'config.json').read_text())
 assert c['coverage_gate']['batch_split_seed']==20260914
 assert c['coverage_gate']['minimum_cells_per_view']==25
 assert c['readout']['primary_k']==10 and c['readout']['secondary_k']==50
 assert c['source']['sha256']=='d3269d9c863d96555eba768b42ab330cc1b57493cd33b4ac8fe7fbab8278f104'
 assert not any(c['governance'].values())
 print('REPLOGLE_BATCH_DISJOINT_PROTOCOL PASS')
if __name__=='__main__': main()
