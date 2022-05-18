from reaktoro import *
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff

T = 320  # K
P = 1e7  # Pa
# T = 273+25
# P = 1e5

label = 'Ca+2'  # Change line 94, third item is looped, rest is in the system but set to 0
output = 'density_aq'  # Can be phase or density_aq

def tersurf(a, b, c, d, line = None, inf_p = None):
    import matplotlib.tri as tri
    """
    :param a: z1
    :param b: z2
    :param c: z3
    :param d: values you want to plot ( e.g. operator values, derivative, hessian,...)
    :param line: in case want to draw trajectory on it
    :param inf_p: inflection point in a given trajectory
    :return:
     """
    z = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])    # transfer matrix
    mt = np.transpose([[1 / 2, 1], [np.sqrt(3) / 2, 0]])    # plot triangle
    # p = np.matmul(z, mt)
    # plt.figure(figsize=(10, 8), dpi=100)
    # plt.plot(p[:, 0], p[:, 1], 'k', 'linewidth', 1.5)
    x = 0.5 - z[:,0] * np.cos(np.pi / 3) + z[:,1] / 2
    y = 0.866 - z[:, 0] * np.sin(np.pi / 3) - z[:, 1] / np.tan(np.pi / 6) / 2
    plt.plot(x, y, 'k', 'linewidth', 1.5)    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) * 0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=3)    # plotting the mesh
    plt.triplot(trimesh, color='navajowhite', linestyle='--', linewidth=0.8)
    plt.ylim([0, 1])
    plt.axis('off')    # translate the data to cords
    x = 0.5 - a * np.cos(np.pi / 3) + b / 2
    y = 0.866 - a * np.sin(np.pi / 3) - b / np.tan(np.pi / 6) / 2    # create a triangulation out of these points
    T = tri.Triangulation(x, y)    # plot the contour
    vmin = min(d) - 0.1   #-10.1
    vmax = max(d) + 0.1   #10.1
    level = np.linspace(vmin, vmax, 101)
    plt.tricontourf(x, y, T.triangles, d, cmap='jet', levels=level)
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], linewidth=1)
    plt.rc('font', size=12)
    cax = plt.axes([0.75, 0.55, 0.055, 0.3])
    plt.colorbar(cax=cax, format='%.3f', label='')
    plt.gcf().text(0.08, 0.1, '$CO_2$', fontsize=20, color='black')
    plt.gcf().text(0.91, 0.1, label, fontsize=20, color='black')
    plt.gcf().text(0.5, 0.8, '$H_2O$', fontsize=20, color='black')
    if line is not None:        # in case, want to draw random trajectories on the ternary diagram
        line = line[:,1:]
        #traj = np.matmul(line, mt)
        #plt.plot(traj[:,0], traj[:,1], '--')
        x = 0.5 - line[:, 0] * np.cos(np.pi / 3) + line[:, 1] / 2
        y = 0.866 - line[:,0] * np.sin(np.pi / 3) - line[:,1] / np.tan(np.pi / 6) / 2
        plt.plot(x,y, '--')
    if inf_p is not None:
        # inf_p = np.matmul(inf_p, mt)
        # plt.scatter(inf_p[:, 0], inf_p[:, 1])
        # translate the data to cords
        x = 0.5 - inf_p[:,0] * np.cos(np.pi / 3) + inf_p[:,1] / 2
        y = 0.866 - inf_p[:,0] * np.sin(np.pi / 3) - inf_p[:,1] / np.tan(np.pi / 6) / 2
        plt.scatter(x, y)
    plt.show()

df = pd.DataFrame(columns=['H2O', 'CO2', label, 'phase', 'density_aq'])
dict_merge = dict()

nop = 20
for j in range(nop):
    print(j)
    for q in range(nop):
        flag = False
        if label == 'ions':
            z_e = [q/nop-0.5*j/nop, 1-q/nop-0.5*j/nop, j/nop, j/nop]
        else:
            z_e = [q/nop-0.5*j/nop, 1-q/nop-0.5*j/nop, j/nop]
        for i in range(len(z_e)):
            if z_e[i] < 0:
                flag = True
        if flag:
            continue

        db = SupcrtDatabase("supcrtbl")
        gas = GaseousPhase("H2O(g) CO2(g)")
        aq = AqueousPhase('H2O(aq) CO2(aq) '+str(label)+' CO3-2 Na+ Cl-')
        # sol = MineralPhase('Calcite')
        # sol2 = MineralPhase('Anhydrite')

        system = ChemicalSystem(db, gas, aq)
        state = ChemicalState(system)
        state.temperature(T, "kelvin")
        state.pressure(P, 'pascal')
        state.set('H2O(aq)', z_e[0], 'mol')
        state.set('CO2(aq)', z_e[1], 'mol')
        state.set(label, z_e[2], 'mol')

        specs = EquilibriumSpecs(system)
        specs.temperature()
        specs.pressure()

        # These lines are the same as charge in PHREEQC, you set open the solution to add more Na+ if the charge isnt 0

        specs.charge()
        specs.openTo("Na+ Cl-")

        conditions = EquilibriumConditions(specs)
        conditions.temperature(state.temperature())
        conditions.pressure(state.pressure())
        conditions.charge(0)  # sets the state to charge using openTo

        solver = EquilibriumSolver(specs)
        solver.solve(state, conditions)

        cq = ChemicalProps(state)
        H2O_aq = cq.speciesAmount('H2O(aq)')
        H2O_g = cq.speciesAmount('H2O(g)')
        H2O = H2O_aq + H2O_g
        CO2_aq = cq.speciesAmount('CO2(aq)')
        CO2_g = cq.speciesAmount('CO2(g)')
        CO2 = CO2_aq + CO2_g
        label_amount = cq.speciesAmount(label)
        other_ion = cq.speciesAmount('CO3-2')
        total_mol = H2O+CO2+label_amount+other_ion
        total_mol_aq = H2O_aq+CO2_aq+label_amount+other_ion

        gas_props: ChemicalPropsPhaseConstRef = cq.phaseProps(0)
        liq_props: ChemicalPropsPhaseConstRef = cq.phaseProps(1)

        # sol_props: ChemicalPropsPhaseConstRef = cq.phaseProps(2)
        # volume_solid = sol_props.volume()
        # density_solid = sol_props.density()

        volume_gas = gas_props.volume()
        volume_aq = liq_props.volume()
        print('og volume',volume_aq)
        density_gas = gas_props.density()
        density_aq = liq_props.density()
        print('og density', density_aq)

        # openTo cancel transport and calculation of ions
        partial_mol_vol_aq = np.zeros(4)
        for i in range(4):
            partial_mol_vol_aq[i] = float(liq_props.speciesStandardVolumes()[i])
        mol_frac_aq = [float(H2O_aq/total_mol_aq), float(CO2_aq/total_mol_aq),
                       float(label_amount/total_mol_aq), float(other_ion/total_mol_aq)]

        volume_aq = total_mol_aq * np.sum(np.multiply(mol_frac_aq, partial_mol_vol_aq))
        volume_tot = volume_aq+volume_gas
        mass_aq = liq_props.mass() - cq.speciesMass('Na+') - cq.speciesMass('Cl-')
        density_aq = mass_aq / volume_aq

        print(volume_aq)
        print(density_aq)

        S_w = volume_aq / volume_tot
        S_g = volume_gas / volume_tot
        # S_s = volume_solid / volume_tot

        L = (density_aq*S_w) / (density_gas*S_g+density_aq*S_w)  # +density_solid*S_s)

        liq_species = liq_props.speciesAmounts()  # returns list of species in mol
        gas_species = gas_props.speciesAmounts()
        # sol_species = sol_props.speciesAmounts()

        # makes the contour plots easier to read
        if density_aq < 0:
            density_aq = -1
        if density_aq > 10000:
            density_aq = 10001
        df_concat = pd.DataFrame({'H2O': [float(liq_species[0]+gas_species[0])],
                        'CO2': [float(liq_species[1]+gas_species[1])],
                        label: [float(liq_species[2])],
                        'phase': [float(L)], 'density_aq': [float(density_aq)]})
        # print(state)
        # print(state.props())
        frames = [df, df_concat]
        df = pd.concat(frames)

df["phase"] = df["phase"].astype(float)
df["density_aq"] = df["density_aq"].astype(float)
fig = px.scatter_ternary(df, a='H2O', b='CO2', c=label, color=output)
fig.show()
print(len(np.array(df["CO2"])))
tersurf(np.array(df["CO2"], dtype=float), np.array(df[label], dtype=float), np.array(df["H2O"], dtype=float),
        np.array(df[output], dtype=float))

# Makes nicer contour plot, but not always works
fig2 = ff.create_ternary_contour(np.array([df["H2O"], df['CO2'], df[label]], dtype=float),
                                 np.array(df[output], dtype=float),
                                 pole_labels=["H2O", "CO2", label],
                                 interp_mode="cartesian",
                                 showscale=True)
fig2.show()
