function test_MC3D ()
  load('MC3Dmex.input.mat');

  % disable_pbar = int64(1);
  use_gpu = false;
  use_alt = true;

  % fprintf('Double version\n');
  % tic;
  % [element_fluence, boundary_exitance, boundary_fluence, simulation_time, seed_used] = MC3Dmex(...
  %   H, HN, BH, r, BCType, BCIntensity, BCLightDirectionType,...
  %   BCLightDirection, BCn, mua, mus, g, n, f,...
  %   phase0, Nphoton, disable_pbar, uint64(rnseed), use_gpu, use_alt);
  % toc;

  fprintf('Single version\n');
  tic;
  [element_fluence, boundary_exitance, boundary_fluence, simulation_time, seed_used] = MC3Dmex(...
    H, HN, BH, single(r), BCType, single(BCIntensity), BCLightDirectionType,...
    single(BCLightDirection), single(BCn), single(mua), single(mus), single(g), single(n), single(f),...
    single(phase0), Nphoton, disable_pbar, uint64(rnseed), use_gpu, use_alt);
  toc;
end
