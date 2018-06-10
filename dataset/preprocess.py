import sys
import os
import json

def get_positions(start_idx, end_idx, length):
	""" Get subj/obj position sequence. """
	return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + list(range(1, length-end_idx))

fi = open(sys.argv[1], 'r')
fo = open(sys.argv[2], 'w')
for line in fi:
	data = json.loads(line.strip())
	ss, se = data['subj_start'], data['subj_end']
	os, oe = data['obj_start'], data['obj_end']
	l = len(data['tokens'])
	data['tokens'][ss:se+1] = ['SUBJ-'+data['subj_type']] * (se-ss+1)
	data['tokens'][os:oe+1] = ['OBJ-'+data['obj_type']] * (oe-os+1)
	data['subj_pst'] = get_positions(ss, se, l)
	data['obj_pst'] = get_positions(os, oe, l)
	fo.write(json.dumps(data) + '\n')
fi.close()
fo.close()

