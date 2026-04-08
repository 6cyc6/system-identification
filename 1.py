from system_identification.excitation_generator_new import generate_constraints
from system_identification.excitation_generator import generate_constraints as generate_constraints_old

n_order = 5
cA, cB = generate_constraints(n_order)
cA_old, cB_old = generate_constraints_old(n_order)

print("cA: ", cA)
print("cB: ", cB)
print("cA_old: ", cA_old)
print("cB_old: ", cB_old)