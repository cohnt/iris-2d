import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.transforms import Affine2D
import alphashape
import cvxpy as cp

np.random.seed(0)

seed_point = None
As = []
bs = []
Cs = []
ds = []
tolerance = 0.00001
limits = [[0, 1], [0, 1]]
max_iters = 10

def gen_obstacles():
	n_points = 200
	alpha = 25.

	points = np.random.random(size=(n_points, 2))
	gen = alphashape.alphasimplices(points)

	tris = []
	for simplex, r in gen:
		if r < 1/alpha:
			tris.append(points[simplex])

	tris.append(np.array([[0, 0], [1, 0], [0.5, -0.5]]))
	tris.append(np.array([[1, 0], [1, 1], [1.5, 0.5]]))
	tris.append(np.array([[1, 1], [0, 1], [0.5, 1.5]]))
	tris.append(np.array([[0, 1], [0, 0], [-0.5, 0.5]]))

	return tris

def draw_ellipse(C, d):
	ts = np.linspace(0, 2*np.pi)
	points = np.array([np.cos(ts), np.sin(ts)])
	points = C @ points + d.reshape(-1,1)
	ax.plot(*(points))

def draw():
	global seed_point, As, bs, Cs, ds
	ax.cla()
	ax.set_xlim(limits[0])
	ax.set_ylim(limits[1])
	ax.set_aspect("equal")
	for tri in tris:
		ax.add_patch(Polygon(tri, color="red"))
	if not (seed_point is None):
		ax.scatter([seed_point[0]], [seed_point[1]])
	if len(Cs) > 0:
		C = Cs[-1]
		d = ds[-1]
		draw_ellipse(C, d)
	if len(As) > 0:
		A = As[-1]
		b = bs[-1]
		for i in range(len(b)):
			w = A[:,i]
			intercept = b[i]
			xx = np.linspace(*limits[0])
			yy = (-w[0] / w[1]) * xx + (intercept / w[1])
			ax.plot(xx, yy, color="blue")
	plt.draw()

def SeparatingHyperplanes(C, d, O):
	C_inv = np.linalg.inv(C)
	C_inv2 = C_inv @ C_inv.T
	O_excluded = []
	O_remaining = O
	ais = []
	bis = []
	while len(O_remaining) > 0:
		print(len(O_remaining))
		xs = []
		dists = []
		for o in O_remaining:
			x, dist = ClosestPointOnObstacle(C, C_inv, d, o)
			xs.append(x)
			dists.append(dist)
		best_idx = np.argmin(dists)
		x_star = xs[best_idx]
		ai, bi = TangentPlane(C, C_inv2, d, x_star)
		ais.append(ai)
		bis.append(bi)
		idx_list = []
		for i, li in enumerate(O_remaining):
			redundant = [np.dot(ai.flatten(), xj) >= bi for xj in li]
			if i == best_idx or np.all(redundant):
				idx_list.append(i)
		for i in reversed(idx_list):
			O_excluded.append(O_remaining[i])
			O_remaining.pop(i)
	A = np.array(ais).T[0]
	b = np.array(bis).reshape(-1,1)
	return (A, b)

def ClosestPointOnObstacle(C, C_inv, d, o):
	v_tildes = C_inv @ (o - d).T
	n = 2
	m = len(o)
	x_tilde = cp.Variable(n)
	w = cp.Variable(m)
	prob = cp.Problem(cp.Minimize(cp.sum_squares(x_tilde)), [
		v_tildes @ w == x_tilde,
		w @ np.ones(m) == 1,
		w >= 0
	])
	prob.solve()
	x_tilde_star = x_tilde.value
	dist = np.sqrt(prob.value) - 1
	x_star = C @ x_tilde_star + d
	return x_star, dist

def TangentPlane(C, C_inv2, d, x_star):
	a = 2 * C_inv2 @ (x_star - d).reshape(-1, 1)
	b = np.dot(a.flatten(), x_star)
	return a, b

def InscribedEllipsoid(A, b):
	n = 2
	C = cp.Variable((n,n), symmetric=True)
	d = cp.Variable(n)
	constraints = [C >> 0]
	constraints += [
		cp.atoms.norm2(ai.T @ C) + (ai.T @ d) <= bi for ai, bi in zip(A.T, b)
	]
	prob = cp.Problem(cp.Maximize(cp.atoms.log_det(C)), constraints)
	prob.solve()
	return C.value, d.value

def optim():
	global As, bs, Cs, ds
	As = []
	bs = []
	Cs = []
	ds = []
	
	C0 = np.eye(2) * 0.01
	Cs.append(C0)
	ds.append(seed_point.copy())
	O = tris

	draw()

	iters = 0

	while True:
		print("Iteration %d" % iters)

		A, b = SeparatingHyperplanes(Cs[-1], ds[-1], O.copy())
		As.append(A)
		bs.append(b)

		draw()
		plt.pause(0.5)

		C, d = InscribedEllipsoid(As[-1], bs[-1])
		Cs.append(C)
		ds.append(d)

		draw()
		plt.pause(0.5)

		iters += 1

		if (np.linalg.det(Cs[-1]) - np.linalg.det(Cs[-2])) / np.linalg.det(Cs[-2]) < tolerance:
			break

		if iters > max_iters:
			break

	print("Done")

def onmousepress(event):
	global A, b, C, d
	global seed_point
	if event.inaxes:
		seed_point = np.array([event.xdata, event.ydata])
		optim()
		draw()

tris = gen_obstacles()

fig, ax = plt.subplots()
fig.canvas.mpl_connect("button_press_event", onmousepress)
draw()
plt.show()