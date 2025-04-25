import yaml
import numpy as np

# Open the YAML file for reading
with open('iiwas_volumn.yml', 'r') as file:
    # Load the YAML data into a Python object
    data = yaml.load(file, Loader=yaml.FullLoader)

volumns = []
for i in range(8):
    key = f'link{i}'
    volumn_i = data[key]
    volumns.append(volumn_i)

proportion = np.array(volumns) / sum(volumns)
print(proportion)
mass = 29.9 * proportion
cad_mass = [5, 4.100238, 3.943457, 4.0, 4.0, 3.2, 2.2, 1.0]
print(mass, "Volumn based")
print(cad_mass, "CAD mass")
print(sum(mass), sum(cad_mass), "volumn vs cad")
print(sum(mass[4: ]), sum(cad_mass[4: ]))