#!/usr/bin/python
import h5py

f = h5py.File('../data/h5_file/duble_human_test/l_115_u.h5', 'r')
#f = h5py.File('../data/h5_file/l_113_e.h5', 'r')

print(list(f.keys()))
print(f['image'])
print(f['lable'])
#for p in f['lable']:
#    print p
print(f['image'][0])

f = h5py.File('../data/h5_file/SingleImageTest.h5', 'r')

print(list(f.keys()))
print(f['rgb'])
print(f['rgb'][0])

'''
print("Singel mean %s" % fs['Mean'])
print("Singel var %s" % fs['Variance'])
print("Singel lable %s" % fs['labels'])
print("Singel rgb %s" % fs['rgb'])

print("Singel rgb0 %s" % fs['rgb'][0,0,0,:])

print("multi mean %s" % f6['Mean'])
print("multi var %s" % f6['Variance'])
print("multi lable %s" % f6['labels'])
print("multi rgb %s" % f6['rgb'])

print("multi mean val %s" % f6['Mean'][()])
#print("multi rgb0 %s" % f6['rgb'][0,0,0,:])
'''
