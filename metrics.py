import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union

def compute_similarity_rankings(embedding, users: Union[list, np.array]):
    
    cos_similarity = cosine_similarity(np.array(embedding))
    
    # get similarity between each pair of users
    similarity_scores = {"User1": [], "User2": [], "Score": []}
    for i,(user,sim_scores) in enumerate(zip(users,cos_similarity)):
        other_users = np.delete(users, i)
        other_sim_scores = np.delete(sim_scores,i)
        for rank,(other_user,sim_score) in enumerate(zip(other_users,other_sim_scores)):
            similarity_scores["User1"].append(user)
            similarity_scores["User2"].append(other_user)
            similarity_scores["Score"].append(sim_score*100)
    similarity_scores = pd.DataFrame.from_dict(similarity_scores)
    
    # rank
    similarity_scores["Ranking"] = [0]*len(similarity_scores)
    for user in users:
        user_df = similarity_scores[similarity_scores["User1"]==user].sort_values("Score",ascending=False)
        for rank,(idx,row) in enumerate(user_df.iterrows()):
             similarity_scores.loc[idx,"Ranking"] = rank+1
                
    return cos_similarity, similarity_scores