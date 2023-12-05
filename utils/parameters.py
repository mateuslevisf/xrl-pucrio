import json

def save_params(args):
    """Saves the parameters used in the experiment to JSON file."""
    save_file = "results/params.json"
    with open(save_file, 'w') as f:
        json.dump(vars(args), f, indent=4)