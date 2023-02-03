# A Multi-Task Model for Long-Term Active Days Prediction
This repository is the official implementation of a multi-task model for long-term Active days prediction(MLADP).

# 1. Abstract 

This paper introduces a new task, user long-term active days prediction, in which short-term behaviors (e.g., 5 or 7 days) are used to predict the long-term active days (e.g., 30 days) of a user on a product or service. This task is significant for internet companies, such as Baidu, as it saves time and resources in gathering information for user retention analysis while providing valuable insights into user behavior and retention. Despite its importance, there is limited research on the user long-term active days prediction task. Existing works in related areas, such as forecasting active users and dropout prediction in MOOCs, rely on historical long-term behavior data to predict future short-term user activity.This study identifies two main challenges in the task: obtaining users' behavior periodic trends based on limited short-term behaviors and modeling personalized trends of different behaviors. To tackle these challenges, we propose a multi-task learning model for long-term active days prediction (MLADP).To solve the first challenge, we firstly integrates inferable periodic time information (such as week, day, month), and propose a periodic-aware attention mechanism to infer user future long-term periodic behavior. To overcome the second challenge, we design a behavior periodic trend prediction task for obtain personalized trends of different behavior. Lastly, we simultaneously train for both the prediction of long-term active days and the prediction of behavior periodic trends. Our experiments on three large datasets demonstrate that the proposed method outperforms existing state-of-the-art methods. The proposed model has been deployed in a real system to analyze user behavior and improve retention.

# 2. Install the Requirments of Experiment
## You can configure the environment in two ways
### Method 1
    conda create -n LADP_Env python=3.6
    conda activate LADP_Env
    pip install torch
    pip install numpy
    pip install pandas
    pip install scikit-learn
    pip install matplotlib
### Method 2
    chmod 777 create_envs.sh
    ./create_envs.sh

# 3. Running
# 3.1 Datasets Selection
Select a dataset you want to include.
There are two kinds of dataset
### Statistics of the datasets
| **Dataset** | **Kwai** | **MOOC** |
|:-----------:|:--------:|:--------:|
|    #User    |  51,709  |  200,904 |
|  #Behavior  |     6    |     7    |
|     #Day    |    30    |    30    |

### The number of users with different 𝐷𝑝
| **Dataset** | **Kwai** | **MOOC** |
|-------------|----------|----------|
|    𝐷𝑝 = 3   |   3,029  | 41,747   |
|    𝐷𝑝 = 5   |   5,172  | 55,059   |
|    𝐷𝑝 = 7   |   8,322  | 66,448   |
|   𝐷𝑝 = 14   |  17,857  | 93,825   |

`data/<dataset>/feature/day_<day>_activity_feature.csv`


Each data set represents the student's activity record of the day in a row.

The record contains the user ID, 
the normalized result after the statistics of the number of activities of 
each behavior and the user portrait.



`data/<dataset>/info/user_info.csv`

Statistics of each student's daily activities in a certain course.



`data/<dataset>/info/user_time_info.csv`


The specific day corresponding to each student in the statistical interval

# 3.2 Training 
We use MOOC as an example of a dataset.

If you want to run the CFIN model to predict the user's activity for 23 days in 7 days
    
    python main.py --model_name 'CFIN' --DataSet 'KDD'  --day 7 --future_day 23
    
If you want to change the random seed of data partition

    python main.py --model_name 'CFIN' --DataSet 'KDD'  --day 7 --future_day 23 --seed 2


# 3.3 Visualization

After our model is run, you can just run `DrawTool.py` to plot the user's predicted activity and real situation in the next N days.

# 4. Results
# 4.1 Experiment Results
Baseline model and Our model achieves the following results on 
Kwai, MOOC, Baidu.

![](.\Figure\Exp_result_1.png)

![](.\Figure\Exp_result_2.png)

![](.\Figure\Exp_result_3.png)

    
    