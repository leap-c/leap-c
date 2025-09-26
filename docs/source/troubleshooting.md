# Troubleshooting usability with VS Code

## Fixing the Code Highlighting in VS Code

Sometimes there is an issue regarding the code analysis in VS Code regarding leap-c . 
Even though you installed leap-c correctly and the code runs properly, e.g., when running a script via the terminal,
VS Code will not recognize the imports and not highlight the imported classes or show their documentation upon hovering.
Possible fixes could be the following:
1. Make sure you installed the Python extension in VS Code (or else there will probably not be any code analysis in the first place).
2. Make sure that you selected the correct interpreter in VS Code: When you have opened a Python file in VS Code, 
you can see your currently used Python version in the lower right corner, e.g., "3.11.13". Click it to select
the correct interpreter, e.g., if you have installed leap-c in a virtual environment named "venv", it is the file
path_to_venv/bin/python . Alternatively, you can use `Ctrl+Shift+P` and search for "Select Interpreter".
3. Use `Ctrl+Shift+P` and search for "Preferences: Open User Settings" and press enter to open a new tab showing your settings.
Right below the search bar click on "Workspace". Now search for "Python Analysis Extra Paths" in the search bar.
In the setting "Python > Analysis: Extra Paths" click on "Add item", then enter the path to the leap-c root folder, e.g.
"~/leap-c".

## Fixing `plt.show()` in when using VS Code to connect to WSL
Matplotlibs `plt.show()` will probably not work when using VS Code to connect to WSL for running leap-c.
To fix it, you can run
```bash
sudo apt install libqt5gui5
```
and then install
`PyQt5` in your python virtual environment using
```bash
pip install pyqt5
```
(This is already contained in some of the package preconfigurations).
Afterwards, restart your VS Code.
If this doesn't work for some reason, you can also save the relevant plots by using `plt.savefig` 
and look at them manually.
