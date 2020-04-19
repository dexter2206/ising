"""Simplest usage of ising.search."""
from ising import search

def main():
    graph = {(0, 0): -1, (1,1):-1, (2, 2): -1, (3, 3): -1}
    solution = search(graph, num_states=10, show_progress=True, use_gpu=False)
    print('Energy | State')
    for energy, state in zip(solution.energies, solution.states):
        print('{0:6.2f} | {1}'.format(energy, state))

if __name__ == '__main__':
    main()
