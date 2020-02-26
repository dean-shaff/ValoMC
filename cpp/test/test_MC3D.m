function test_MC3D ()
  load('MC3Dmex.input.mat');

  % disable_pbar = int64(1);
  use_gpu = false;

  [element_fluence, boundary_exitance, boundary_fluence, simulation_time, seed_used] = MC3Dmex(...
    H, HN, BH, r, BCType, BCIntensity, BCLightDirectionType,...
    BCLightDirection, BCn, mua, mus, g, n, f,...
    phase0, Nphoton, disable_pbar, uint64(rnseed), use_gpu);

end
