import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from constraints import Constraint


class Sampler:
    def __init__(self, filename, max_points, max_time):
        self.constraint = Constraint(filename)
        self.n_dim = self.constraint.n_dim
        self.example = self.constraint.example
        self.max_points = max_points
        self.max_time = max_time

    def test_inside(self, x):
        if np.any(np.less(x, 0.)):
            return False
        if np.any(np.greater(x, 1.)):
            return False
        return self.constraint.apply(x)

    def line_search(self, x, direction):
        tiny = 1e-6
        low, high = 0., np.sqrt(self.n_dim) + tiny
        for _ in range(20):
            middle = 0.5 * (low + high)
            inside = self.test_inside(x + middle * direction)
            if inside:
                low = middle
            else:
                high = middle
        return low

    def random_direction(self):
        vector = np.random.normal(size=self.n_dim)
        return vector / np.linalg.norm(vector)

    def hit_and_run(self):
        assert self.test_inside(self.example)

        points = [np.array(self.example)]
        start_time = time.time()
        while len(points) < self.max_points and time.time() - start_time < self.max_time:
            direction = self.random_direction()

            factor = self.line_search(points[-1], direction)

            if factor > 0.:
                new_point = points[-1] + np.random.uniform() * factor * direction
                if self.test_inside(new_point):
                    points.append(new_point)
        return points


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Sampler for polytope")
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile',type=str)
    parser.add_argument('n_results', type=int)
    args = parser.parse_args()

    max_points = 100000
    max_time = 4.9 * 60.  # just under 5 minutes

    sampler = Sampler(args.infile, max_points, max_time)

    points = sampler.hit_and_run()

    points = np.stack(points)
    points = points[np.random.choice(len(points), size=args.n_results)]
    np.savetxt(args.outfile, points)

    x, y = points.T[0:2]

    plt.scatter(x, y, s=1)
    plt.show()