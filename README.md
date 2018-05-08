# Speech DB Engine
### A project that simplifies the usability of the predominant speech databases.

## Getting Started

### Download from Pypi

```bash
pip install speechdb
```

### General Syntax

```python3
from speechdb import timit
timit1 = timit.Timit(path_to_timit_database,path_to_core_test_csv)

# yType: {'PHN','DLCT','SPKR'} (default='PHN')
# yVals: 'All' or ['sh',...] or ['DR1',...] or ['FDAB0',...] (default='All')
# dataset: {'test','train','coretest') (default='train')
y,x = timit1.read_db(yType, yVals, dataset)

ex.

y,x = timit1.read_db('PHN',['sh'],'test')

```
