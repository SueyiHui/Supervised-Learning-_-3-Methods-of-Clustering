# Supervised-Learning-_-3-Methods-of-Clustering
You work at a local jewelry store. You’ve recently been promoted and the store owner asked you to better understand your customers. Using some sneaky magic (and the help of Environics!), you’ve managed to collect some useful features for a subset of your customers: age, income, spending score, and savings (i.e., how much savings they have in their bank account). Use these features to segment your customers and create customer personas.

## **3 Method Used for Clustering**
Model | Parameters | Silhouette | Notes
----- | ---------- | ---------- | -----
KMEANS | K = 5	0.805 using Euclidean | Inertia = 66.511 | Conducted Elbow Method to conclude when K = 5, both inertias plot and Silhouettes plot reach to their optimal result. 
DBSCAN | MinPts = 3, Eps = 0.5	,	Euclidean| 0.805 | **1**. Conducted Elbow Method to conclude when eps is 0.3 and onwards seems to perform a better Silhouette Score. **2**. Dive deep by established Parameter Exploration Method to conclude the best parameter is eps = 0.5, minpts = 3 and k =5.
Hierarchical | Linkage = complete, correlation |0.951	| **1**. Compared the outcomes of Linkage is “ward” and Metric is “Euclidean” (0.805) V.S. Linkage is “complete” and Metric is “correlation”(0.951). **2**. Conducted Visual Features for each cluster for hyper parameters, as the result, both clusters for each method looks the same.

## **Cluster Group Identified and Personas"**

**Cluster 1 – YOLO Believers**

**Cluster Motto: “Spend While You Can, Because You Only Live Once”**

**Segment Descriptions:** This group of clients are the fresh young industry elites, with an average age of 24, who do not care about savings in the future but the excitement of having “high-end” products and keeping up with the trend. They are the highest earning group with highest spending score, however, with the lowest saving amount. Even though this group could be considered as the perfect marketing target for jewelry stores, we still need to be cautious due to its lack of ability to save.

**Cluster 2 – Fashion Chaser**

**Cluster Motto: “Fashion, Never too Late to Chase Them”**

**Segment Descriptions:** This group of clients are the middle-aged individuals who are avid jewelry purchasers with an above average spending score of 0.77. Unlike customers in cluster 1, clients in this segment are earning below average income and are much older. However, like cluster 1, clients in this sector are passionate about jewelries and do not care too much about their savings in the bank.

**Cluster 3 – Regain-Youth Grammas**

**Cluster Motto: “I’ve Been Working Hard to Reach My Saving Goals, Now It Is Time to Indulge Them”**

**Segment Descriptions:** This group of clients are the individuals who work extremely hard in their careers and achieved the highest savings amounts among all the clusters. Although their income is below average due to retirement, they still manage to wisely spend money on jewelries with a lower than average spending score of 0.33. 

**Cluster 4 – Buying-Jewelry-to-Survive Husbands**

**Cluster Motto: “I Am Forced Here for A 60 Year-Anniversary Gift, Hopefully This Will Not Happen Twice”**

**Segment Descriptions:** This group of clients are in their 80s with above average incomes. Even though with their excellent income revenue and saving amount, given by the extremely low spending score, this group of people is reluctant to spend ample of money in purchasing jewelries. 

**Cluster 5 – Reasonable Spenders**

**Cluster Motto: “Though There are Lots of Temptations, I Know Exactly What I Want”**

**Segment Descriptions:** This group of clients are individuals in middle level management position with above average incomes and savings. Compare to Cluster 4, this group of customers are much younger and more frequently spending money on jewelries. With spending score blow than average, it indicates that this segment likes fashion, but with rational spending behaviors. 
