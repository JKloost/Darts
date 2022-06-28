from reaktoro import *
import numpy as np
import os
import matplotlib.pyplot as plt
db = PhreeqcDatabase("phreeqc.dat")

solution = AqueousPhase(speciate("H O C Ca Na Mg Cl"))
solution.setActivityModel(chain(
    ActivityModelHKF(),
    ActivityModelDrummond("CO2")
))

exchange_species = "NaX CaX2 MgX2"
exchange = IonExchangePhase(exchange_species)
exchange.setActivityModel(ActivityModelIonExchangeGainesThomas())

system = ChemicalSystem(db, solution, exchange)

T = 25.0 # temperature in celsius
P = 1.0  # pressure in bar

state = ChemicalState(system)
state.setTemperature(T, "celsius")
state.setPressure(P, "bar")
state.setSpeciesMass("H2O", 1.e6, "kg")
state.setSpeciesAmount("Na+", 1.10, "kmol")
state.setSpeciesAmount("Mg+2", 0.48, "kmol")
state.setSpeciesAmount("Ca+2", 1.90, "kmol")
# Set the number of exchange assuming that it is completely occupied by sodium
state.setSpeciesAmount("NaX", 0.06, "mol")

# Define equilibrium solver and equilibrate given initial state with input conditions
solver = EquilibriumSolver(system)
solver.solve(state)
print(state)

aqprops = AqueousProps(state)
print("I  = %f mol/kgw" % aqprops.ionicStrength()[0])
print("pH = %f" % aqprops.pH()[0])
print("pE = %f" % aqprops.pE()[0])

chemprops = ChemicalProps(state)
exprops = IonExchangeProps(chemprops)
print(exprops)