# handle-missing-values    

# Package Description :
Python package for Detecting and Handling missing values by visualizing and applying different algorithms.
# Motivation :   
This is a part of project - III made for UCS633 - Data analytics and visualization at TIET.     
@Author : Sourav Kumar    
@Roll no. : 101883068    
# Algorithm :       
* **Row removal** / **Column removal** : It removes rows or columns (based on arguments) with missing values / NaN.   
Python’s pandas library provides a function to remove rows or columns from a dataframe which contain missing values or NaN.   
It will remove all the rows which had any missing value. It will not modify the original dataframe, it just returns a copy with modified contents.   
Default value of ‘how’ argument in dropna() is ‘any’ & for ‘axis’ argument it is 0. It means if we don’t pass any argument in dropna() then still it will delete all the rows with any NaN.      
* 
### Getting started Locally :  
> Run On Terminal       
```python -m outlier.outlier inputFilePath outputFilePath z_score```     
or
```python -m outlier.outlier inputFilePath outputFilePath iqr```       
ex. python -m outlier outlier C:/Users/DELL/Desktop/train.csv C:/Users/DELL/Desktop/output.csv z_score     

> Run In IDLE   
```from outlier import outlier```   
```o = outlier.outlier(inputFilePath, outputFilePath)```     
```o.outlier_main('z_score')```
or    
```o.outlier_main('iqr')```     

> Run on Jupyter   
Open terminal (cmd)   
```jupyter notebook```   
Create a new python3 file.     
```from outlier import outlier```   
```o = outlier.outlier(inputFilePath, outputFilePath)```
```o.outlier_main('z_score')```
or    
```o.outlier_main('iqr')```       

* NOTE : ```outlier_main()``` doesn't necessarily require any ```method``` argument , if no argument is provided, it uses ```z_score``` by default as the algorithm for removal of outliers from the dataset.    
* The algorithm only reports missing data containing columns and not handles them, it assumes that it has been handled already.   
Also in case of z-score method, it will not affect much, but it may be possible to give wrong output in case of IQR if missing values are found.    
### OUTPUT :
After analysing and visualizing every possible algorithm against metrics (accuracy, log_loss, recall, precision), The best algorithm is applied for imputing the missing values in the original dataset.    
Also , the final dataframe will be written to the output file path you provided.
 
![output result on jupyter]()
![output result on idle]()
![output result on cmd]() 

# TESTING : 
* The package has been extensively tested on various datasets consisting varied types of expected and unexpected input data and any preprocessing , if required has been taken care of.

