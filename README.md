This paper introduces a new task, user long-term active days prediction, in which short-term behaviors (e.g., 5 or 7 days) are used to predict the long-term active days (e.g., 30 days) of a user on a product or service. 
This task is significant for internet companies, such as Baidu, as it saves time and resources in gathering information for user retention analysis while providing valuable insights into user behavior and retention. 
Despite its importance, there is limited research on the user long-term active days prediction task. Existing works in related areas, such as forecasting active users and dropout prediction in MOOCs, rely on historical long-term behavior data to predict future short-term user activity.  
This study identifies two main challenges in the task: obtaining users' behavior periodic trends based on limited short-term behaviors and modeling personalized trends of different behaviors. 
To tackle these challenges, we propose a multi-task learning model for long-term active days prediction (MLADP).
To solve the first challenge, we firstly integrates inferable periodic time information (such as week, day, month), and propose a periodic-aware attention mechanism to infer user future long-term periodic behavior. 
To overcome the second challenge, we design a behavior periodic trend prediction task for obtain personalized trends of different behavior. 
Lastly, we simultaneously train for both the prediction of long-term active days and the prediction of behavior periodic trends.
Our experiments on three large datasets demonstrate that the proposed method outperforms existing state-of-the-art methods. The proposed model has been deployed in a real system to analyze user behavior and improve retention.
