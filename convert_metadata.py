import json

cities = ['aschaffenburg','augsburg','bamberg','bayreuth','erlangen','fuerth','ingolstadt','kempten','landshut','muenchen','neuulm','nuernberg','regensburg','rosenheim','schweinfurt','wuerzburg']

for city in cities:
    filepath =r'/home/abasu/gnn_predicting_effects_of_traffic_policies/data/inductive_data/training_data/kreisfreistadt/'+city+'/metadata.json'
# Load the original file
    with open(filepath, "r") as f:
        data = json.load(f)

    # Define the mapping from old to new base path
    old_base = "/home/abasu/gnn_predicting_effects_of_traffic_policies/data/inductive_data/training_data/kreisfreistadt/"
    new_base = "/dss/dssfs03/pn39mu/pn39mu-dss-0000/bavaria_simulations/inductive_data/training_data/kreisfreistadt/"

    # Update the paths
    data["path"] = [
        path.replace(old_base, new_base) for path in data["path"]
    ]

    # Save the updated file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Updated {city} metadata")
