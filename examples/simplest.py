"""Simplest usage of ising.search."""
from ising import search

def main():
    graph = {(1,1):-1, (1,2):-0.2}
    solution = search(graph, num_states=4)
    print('Energy | State')
    for energy, state in zip(solution.energies, solution.states):
        print('{0:6.2f} | {1}'.format(energy, state))

if __name__ == '__main__':
    main()
