# Speech DB Engine
### A project that simplifies the usability of the predominant speech databases.

## Usage

### Download from Pypi


### Syntax

```python3
from timit import Timit
timit = Timit(path_to_timit_database,path_to_core_test_csv)

# yType: {'PHN','DLCT','SPKR'} (default='PHN')
# yVals: 'All' or ['sh',...] (default='All')
# dataset: {'test','train','coretest') (default='train')
y,x = timit.read_db(yType, yVals, dataset)

```
