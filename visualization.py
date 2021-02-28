import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from typing import Tuple, Union
from sklearn.decomposition import PCA
from ast import literal_eval

def plot_embedding(
    embedding: np.array, 
    labels: Union[np.array, list], 
    dim: str="2d",  
    title: str="",
    xlims: Tuple[float,float]=None,
    ylims: Tuple[float,float]=None,
    zlims: Tuple[float,float]=None,
    figsize: Tuple[int,int]=None,
    include_labels: bool=True,
    highlight_labels: bool=False
):
    
    if len(embedding) != len(labels):
        print("Number of labels does not match first dimension of embedding, returning None")
        return None
    
    if not include_labels:
        labels = [""]*len(labels)
    
    if dim == "2d":
        
        if len(labels) < 2:
            print("Number of data points is less than 2, can't be plotted in 2D, returning None")
            return None
        
        if highlight_labels:
            colors = ["tab:red" if len(l)>0 else "tab:blue" for l in labels]
            alphas = [1.0 if len(l)>0 else 0.5 for l in labels]
        else:
            colors = ["tab:blue"]*len(labels)
            alphas = [1.0]*len(labels)

        # project to 2 dimensions
        embed_2d = PCA(n_components=2).fit_transform(embedding)

        # plot
        figsize = (9,6) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        for i, (c,a,label) in enumerate(zip(colors,alphas,labels)):
            plt.scatter(embed_2d[i, 0], embed_2d[i, 1], c=c, alpha=a)
            plt.annotate(label, xy=(embed_2d[i, 0], embed_2d[i, 1]))
        plt.xlabel("dimension 1",fontsize=16)
        plt.ylabel("dimension 2",fontsize=16)
        plt.title(title)
        if xlims is not None:
            plt.xlim(xlims)
        if ylims is not None:
            plt.ylim(ylims)
        plt.show()
        
    elif dim == "3d":
        
        if len(labels) < 3:
            print("Number of data points is less than 3, can't be plotted in 3D, returning None")
            return None
        
        if highlight_labels:
            colors = ["tab:red" if len(l)>0 else "tab:blue" for l in labels]
            alphas = [1.0 if len(l)>0 else 0.5 for l in labels]
        else:
            colors = ["tab:blue"]*len(labels)
            alphas = [1.0]*len(labels)
            
        # project to 3 dimensions
        embed_3d = PCA(n_components=3).fit_transform(embedding)

        # plot
        figsize = (9,6) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
        for i, (c,a,label) in enumerate(zip(colors,alphas,labels)):
            ax.scatter(embed_3d[i, 0], embed_3d[i, 1], embed_3d[i, 2], c=c, alpha=a)
            ax.text(embed_3d[i, 0], embed_3d[i, 1], embed_3d[i, 2], label)
        ax.set_xlabel("dimension 1",fontsize=16)
        ax.set_ylabel("dimension 2",fontsize=16)
        ax.set_zlabel("dimension 3",fontsize=16)
        ax.set_title(title)
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
        if zlims is not None:
            ax.set_zlim(zlims)
        plt.show()
        
    else:
        print(f"Dimension {dim} not supported, use either '2d' or '3d'")
        return None
    
    return fig

def get_shared_responses(user1: str, user2: str, category: str):
    response_data = pd.read_csv("data/response_data.csv")
    
    df = response_data[response_data["Category"]==category]
    
    responses = df[
        (df["Name"]==user1)|
        (df["Name"]==user2)
    ].groupby("Response",as_index=False).count()
    
    return responses[responses["Category"]>1]["Response"].tolist()

def compare_users(user1: str, user2: str, columns: list, sim_score: float):
    users = [user1, user2]
    locs = np.linspace(0.9,0.3,len(columns))
    
    clean_data = pd.read_csv("data/clean_data.csv")
    response_data = pd.read_csv("data/response_data.csv")

    fig,axes = plt.subplots(1,3,figsize=(20,10))
    for i,(user,ax) in enumerate(zip(users,np.ravel(axes)[:-1])):

        user_df = clean_data[clean_data["Name"]==user]

        # name
        n_responses = len(response_data[response_data["Name"]==user])
        ax.text(0.5,0.95,f"{user} -- {n_responses} responses",fontsize=16)

        for col,loc in zip(columns,locs):
            col_data = user_df[col].tolist()[0]
            if isinstance(col_data,str) and col_data!="nan":
                responses = literal_eval(col_data)
                shared_data = get_shared_responses(user1,user2,col)
                responses = [r if r in shared_data else r for r in responses]
            else:
                responses = []
            col_data = ", ".join(responses)
            ax.text(0.1,loc,col,fontsize=12)
            txt = ax.text(0.1,loc-0.075,col_data,wrap=True)
            txt._get_wrap_line_width = lambda : 350.
            ax.set_axis_off()

    # title - similarity score
    axes[-1].text(0.3,0.95,f"Similarity -- {np.round(sim_score,2)}%",fontsize=16)

    for col,loc in zip(columns,locs):
        shared_data = get_shared_responses(user1,user2,col)
        n_responses_per_col = len(
            response_data[
                (response_data["Category"]==col)&
                ((response_data["Name"]==user1)|
                (response_data["Name"]==user2))
            ]["Response"].unique()
        )
        axes[-1].text(0.3,loc,col,fontsize=12)
        axes[-1].text(0.3,loc-0.05,f"{len(shared_data)}/{n_responses_per_col} shared")
    axes[-1].set_axis_off()

    plt.tight_layout()
    plt.show()
    
    return fig