# Machine Learning Surrogates for Agent-Based Models in Transportation Policy Analysis

Effective traffic policies are crucial for managing congestion and reducing emissions. Agent-based transportation models (ABMs) offer a detailed analysis of how these policies affect travel behaviour at a granular level. However, computational constraints limit the number of scenarios that can be tested with ABMs and therefore their ability to find optimal policy settings.In this proof-of-concept study, we propose a machine learning (ML)-based surrogate model to efficiently explore this vast solution space. Leveraging a Graph Neural Network (GNN), the model predicts the effects of traffic policies on the road network at the link level.  We implement our approach in a large-scale MATSim simulation of Paris, France, covering over 30,000 road segments and 10,000 simulations, applying a policy involving capacity reduction on main roads. The ML surrogate model achieves an overall R2 of 0.76, with significantly lower Mean Squared Error and Mean Absolute Error than naive baselines. On primary roads where the policy applies, it reaches an R2 of 0.95. This study therefore shows that GNNs can act as surrogates for complex agent-based transportation models with the potential to enable large-scale policy optimization, helping urban planners explore a broader range of interventions more efficiently.

Read more about the project in our [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5182100#).

## Getting Started

To create a virtual environment and install the required packages (using conda), run the following:

```conda env create -f traffic-gnn.yml```

Then, activate the environment:

```conda activate traffic-gnn```

We use Python 3.10.13 along with CUDA 12.7 and CuDNN 8.7 for this project.