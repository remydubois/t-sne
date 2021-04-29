import matplotlib.pyplot as plt
import numpy as np
from timeit import Timer
from tsne.neighbors import QuadTree
from tsne.conditional import compute_joint_gaussian
from tsne.barnes_hut_gradients import compute_barnes_hut_gradients
from tsne.gradients import exact_kl_gradient


def test_resultant_speed():
    X = np.random.uniform(0, 1.0, (1_000, 3))
    y = np.arange(len(X)).astype(np.float64)[:, None].repeat(2, axis=1) / len(X)

    P = compute_joint_gaussian(X)
    tree = QuadTree(y, 16)
    _ = tree.compute_many_resultants(0.1)
    _ = compute_barnes_hut_gradients(P, y, theta=0.1)
    _ = exact_kl_gradient(P, y)

    tree_building_time = Timer(lambda: QuadTree(y, 16)).timeit(10) / 10 * 1000
    tree_query_time = Timer(lambda: tree.compute_many_resultants(0.1)).timeit(10) / 10 * 1000
    bh_time = Timer(lambda: compute_barnes_hut_gradients(P, y, theta=0.1)).timeit(10) / 10 * 1000
    exact_time = Timer(lambda: exact_kl_gradient(P, y)).timeit(10) / 10 * 1000

    np.testing.assert_array_less(
        tree_building_time + tree_query_time,
        exact_time,
        err_msg="Barnes-Hut tree proces slower than exact",
    )
    np.testing.assert_array_less(
        bh_time, exact_time, err_msg="Barnes-Hut tree proces slower than exact"
    )
    print(
        f"Gradient step speed test: {tree_building_time:.1f} ms (Tree building) + "
        f" {tree_query_time:.1f} ms (tree querying time)"
        f"\n-> {bh_time:.1f} ms (complete Barnes-Hut step)"
        f" versus {exact_time:.1f} ms (exact kl step)"
    )


def test_resultant_error():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1.0, (500, 3))
    y = rng.uniform(0, 1.0, (500, 2))

    P = compute_joint_gaussian(X)
    tree = QuadTree(y, 16)

    zs = []
    mae = []
    angles = np.arange(0, 10.0, 0.1)
    for angle in angles:
        zF, z = tree.compute_many_resultants(angle)
        if angle == 0.0:
            mae.append(0.0)
            zs.append(0.0)
            ref = zF / z
            zref = z
        else:
            mae.append(np.abs(ref - zF / z).mean())
            zs.append((z - zref) / zref * 100)

    with plt.xkcd():
        f, ax = plt.subplots()
        ax.plot(angles, zs)
        ax2 = ax.twinx()
        ax.set_ylabel("Z (% error)")
        ax2.plot(angles, mae, color="orange")
        ax2.set_ylabel("Frep MAE")
        ax.set_xlabel("Angle")
        # ax.axhline(zs[0])
        f.show()
