# DEA-DYPN

Feel free to cite this work: 
Wang, R., Liu, W., Li, K., Zhang, T., Wang, L., & Xu, X. (2024). Solving Orienteering Problems by Hybridizing Evolutionary Algorithm and Deep Reinforcement Learning. IEEE Transactions on Artificial Intelligence.

## Requirements:

* Python 3.8
* pytorch
* matplotlib

# To Train the DYPN model

Run by calling ```python trainer.py```

To restore a checkpoint, you must specify the path to a folder that has "actor.pt" and "critic.pt" checkpoints. 

# To Generate the Test Data

Run by calling ```python generate_data.py```

# To Run the DEA-DYPN for Test

Run by calling ```python DEA-DYPN-main.py```


# To Conduct the ablation experiments

Set the following parameters in the ```DEA-DYPN-main.py``` file:

Do not use the greedy initialization: ablation_value_list = [1] 

Do not use the restart mechanism: ablation_value_list = [2]

Do not use the fitness sharing selection: ablation_value_list = [3]

Do not use all these mechanisms: ablation_value_list = [1, 2, 3]

Use all these mechanisms: ablation_value_list = [None]

Then run by calling ```python DEA-DYPN-main.py```


