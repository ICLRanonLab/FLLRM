# FLLRM
The main scripts are in the bin file. If you want to test the simulated data, please run the normal_asNoise_fitmain.py. In this script, the  back ground noise is normal distribution with the fixed meam and sigma and the true pattern is LLR-K(k=1 defaultly) data. For example,
the true pattern with size 500, error 0 and mean 1(sd)
```python
python  normal_asNoise_fitmain.py --xn 500 --mean 10 --errorStdBias 0 # the mean or error will be devided by 10 in the script.
```
After running, you can get the accuracy of detecting the true pattern and the back ground noise.
If you want to test the real data, please download the data from our anonymous google drive  put them in the data file. For example, 
```python
python GSEData_fitMain.py --gseId GSE72056
```
After running you can get the sub-matrices saved as a npy file.
