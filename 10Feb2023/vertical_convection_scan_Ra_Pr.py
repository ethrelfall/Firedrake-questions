# this is example from https://www.firedrakeproject.org/demos/rayleigh-benard.py
# with small changes by Ed Threlfall, 30 December 2022


from firedrake import *
import math

M = Mesh("refined_unit_square.msh")  # 40*40, stretched in x and y warp factor w=12

V = VectorFunctionSpace(M, "CG", 3)  # CG2, CG1 for pressure is Taylor-Hood
W = FunctionSpace(M, "CG", 2)
Q = FunctionSpace(M, "CG", 2)
Z = V * W * Q

upT = Function(Z)
u, p, T = split(upT)
v, q, S = TestFunctions(Z)

x, y = SpatialCoordinate(M)

Ra = Constant(1e6)
Pr = Constant(1.0)

g = Constant((0, 1))  # Ed: changed sign here because the buoyancy force is UP

F = (
    inner(grad(u), grad(v))*dx
    + inner(dot(grad(u), u), v)*dx
    - inner(p, div(v))*dx
    - (Ra/Pr)*inner(T*g, v)*dx
    + inner(div(u), q)*dx
    + inner(dot(grad(T), u), S)*dx
    + 1/Pr * inner(grad(T), grad(S))*dx
)

bcs = [
    DirichletBC(Z.sub(0), Constant((0, 0)), (11, 12, 13, 14)),
    DirichletBC(Z.sub(2), Constant(1.0), (14,)),
    DirichletBC(Z.sub(2), Constant(0.0), (12,))
]

# Like Navier-Stokes, the pressure is only defined up to a constant.::

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True), Z.sub(2)])


# First off, we'll solve the full system using a direct solver.

from firedrake.petsc import PETSc

f = open("convection_outputs.txt", "w+")
f.write("log10Ra, log10Pr, log10Nu\n")

for j in range (1,5):
   Pr.assign(10**(-2+(j-1)/2))

   print("starting log10 Pr = "+str(float(math.log10(Pr)))+" series ...")
   # reset fields: THIS ISN'T WORKING
   u = interpolate(as_vector([0.0*x, 0.0*x]), V)  #DO I HAVE TO USE constant(...) IF x OR y IS NOT IN THE EXPRESSION?
   p = interpolate(0.0*x, W)
   T = interpolate(1.0-x ,Q)
   # does this help: NO
   v = interpolate(as_vector([0.0*x, 0.0*x]), V)
   q = interpolate(0.0*x, W)
   S = interpolate(1.0-x, Q)

   for i in range (1,15):
      Ra.assign(10**(i/2))
      print("starting log10 Ra="+str(float(math.log10(Ra)))+" run ...")
      try:
         solve(F == 0, upT, bcs=bcs, nullspace=nullspace,
               solver_parameters={"mat_type": "aij",
                                  "snes_monitor": None,
                                  "ksp_type": "gmres",
                                  "pc_type": "lu",
                                  "pc_factor_mat_solver_type": "mumps"})
      except:
         break

      normL = Function(V)
      normL = Constant((-1.0,0.0))
      fluxL = assemble(inner(normL, grad(T))*ds(14))  
      normR = Function(V)
      normR = Constant((1.0,0.0))
      fluxR = assemble(inner(normR, grad(T))*ds(12))   
      f.write(str(float(math.log10(Ra)))+", "+str(float(math.log10(Pr)))+", "+str(math.log10(abs(0.5*(fluxL-fluxR))))+"\n")
      print("finished run with log10 Ra "+str(float(math.log10(Ra)))+" and log10 Pr "+str(float(math.log10(Pr))))

f.close()

# do output here if desired
#u, p, T = upT.split()
#u.rename("Velocity")
#p.rename("Pressure")
#T.rename("Temperature")
#File("benard_mod.pvd").write(u, p , T)

print('finished - quitting.')
quit()





