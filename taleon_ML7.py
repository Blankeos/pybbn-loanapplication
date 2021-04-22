import pandas as pd # for data manipulation 
import matplotlib.pyplot as plt # for drawing graphs
import networkx as nx # for drawing graphs

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Import dataset
df = pd.read_csv('loan_application.csv')
print(df)

# PyBBN Stuff | Reference: https://towardsdatascience.com/bbn-bayesian-belief-networks-how-to-build-them-effectively-in-python-6b7f93435bba
# bbn strucutre:
# first I created a DAG considering the variables.
# age is the parent of hasJob.
# hasJob is the parent of ownHouse.
# hasJob is the parent of creditRating
# creditRating is the parent of class (loan approval)

# variables
var_age = Variable(0, 'Age', ['young','middle','old'])
var_hasJob = Variable(1, 'Has_Job', ['False', 'True'])
var_ownHouse = Variable(2, 'Own_House', ['False','True'])
var_creditRating = Variable(3, 'Credit_Rating', ['fair','good','excellent'])
var_class = Variable(4, 'Class', ['No','Yes']) # approved or not

# This function helps to calculate probabily distribution, which goes into BBN (note, can handle upto 2 parents)
def probs(data, child, parent1=None, parent2=None):
    # Initialize empty list
    prob=[]
    if parent1==None:
        # Calculate probabilities
        prob=data[child].value_counts(normalize=True, sort=False).sort_index().tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Work out the bands present in the parent variable
                bands=df[parent1].value_counts(sort=False).sort_index().index.tolist()
                # Caclucate probabilities
                for val in bands:
                    temp=data[data[parent1]==val][child].value_counts(normalize=True).sort_index().tolist()
                    prob=prob+temp
            else:    
                # Work out the bands present in the parent variable
                bands1=df[parent1].value_counts(sort=False).sort_index().index.tolist()
                bands2=df[parent2].value_counts(sort=False).sort_index().index.tolist()
                # Caclucate probabilities
                for val1 in bands1:
                    for val2 in bands2:
                        temp=data[(data[parent1]==val1) & (data[parent2]==val2)][child].value_counts(normalize=True).sort_index().tolist()
                        prob=prob+temp
    else: print("Error in Probability Frequency Calculations")
    return prob  

cpt_age = probs(df, child='Age')
cpt_hasJob = probs(df, child='Has_Job', parent1='Age')
cpt_ownHouse = probs(df, child='Own_House', parent1='Has_Job')
cpt_creditRating = probs(df, child='Credit_Rating', parent1='Has_Job', parent2='Own_House')
cpt_class = cpt_class = probs(df, child='Class', parent1='Credit_Rating')

node_age = BbnNode(var_age, cpt_age)
node_hasJob = BbnNode(var_hasJob, cpt_hasJob)
node_ownHouse = BbnNode(var_ownHouse, cpt_ownHouse)
node_creditRating = BbnNode(var_creditRating, cpt_creditRating)
node_class = BbnNode(var_class, cpt_class)

bbn = Bbn() \
    .add_node(node_age) \
    .add_node(node_hasJob) \
    .add_node(node_ownHouse) \
    .add_node(node_creditRating) \
    .add_node(node_class) \
    .add_edge(Edge(node_age, node_hasJob, EdgeType.DIRECTED)) \
    .add_edge(Edge(node_hasJob, node_ownHouse, EdgeType.DIRECTED)) \
    .add_edge(Edge(node_hasJob, node_creditRating, EdgeType.DIRECTED)) \
    .add_edge(Edge(node_ownHouse, node_creditRating, EdgeType.DIRECTED)) \
    .add_edge(Edge(node_creditRating, node_class, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn) # This is where the bug is, I have no idea how to fix it.

# Set options for graph looks
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5,}
    
# Generate graph
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, **options)

# Update margins and print the graph
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()