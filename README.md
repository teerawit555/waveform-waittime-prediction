### Verify available Python versions
```powershell
py -0
```
### Create a virtual environment using Python 3.11
```powershell
py -V:3.11 -m venv venv311
```
### Enable script execution (Required for PowerShell activation)
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
### Activate the virtual environment
```powershell
.\venv311\Scripts\Activate.ps1
```
### Verify the current Python version
```powershell
python --version
```
### Install dependencies
```powershell
pip install -r requirements.txt
```