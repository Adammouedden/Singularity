# Libaries:

Transformers is a submodule of Singularity. Meaning it is another repo cloned inside of our main repo. 

Having Transformers as a submodule allows the commits to be tracked separately. To receive new changes, run:
```git submodule update```


## Developer Pipeline:

For first time cloning of the Singularity repo, run the following command:
```git clone --recurse-submodules <repo-url>```

For those who already have Singularity cloned but have never initialized the submodule, run the following:
```git pull```
```git submodule update --init --recursive```

For those who already have Singularity cloned and have already initialized the submodule:
```git pull --recurse-submodules```

To automaticaly recursively pull submodule changes:
```git config submodule.recurse true```
