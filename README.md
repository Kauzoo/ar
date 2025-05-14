# Augmented Reality

- Recommended godot version: `godot-stable_v4.4`
- Recommended opencv version: `opencv-4.9.0`


### 2. Installation
Switch into godot-cpp directory and initialize submodule.
```
cd godot-cpp
git submodule update --init
cd ..
```

##### 2.1 Linux


### 3. Clion
See godot *[Configuring an IDE](https://docs.godotengine.org/en/stable/contributing/development/configuring_an_ide/clion.html)* for instructions.  
Inside project root run:
```
scons compiledb=yes
```
to create compilation database file. Afterwards create devbuild target:
```
scons platform=<platform> dev_build=yes
```
To turn on code analysis: In Clion right click on `compile_command.json` and select `Load as project`
