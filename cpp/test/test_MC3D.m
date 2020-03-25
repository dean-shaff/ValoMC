 function test_MC3D (n_photons)
  if ~exist('n_photons', 'var')
    n_photons=1e6;
  end
  n_photons=int64(n_photons);
  load('MC3Dmex.input.mat');
  fprintf('Testing with %d photons\n', n_photons);

  disable_pbar = int64(1);
  single_res = zeros(1, 2);
  double_res = zeros(1, 2);


  for use_alt = 0:1
    if ~use_alt
      text='Reference';
    else
      text='Alternate';
    end

    fprintf('%s CPU double precision version\n', text);
    t_start = tic;
    [element_fluence, boundary_exitance, boundary_fluence, simulation_time, seed_used] = MC3Dmex(...
      H, HN, BH, r, BCType, BCIntensity, BCLightDirectionType,...
      BCLightDirection, BCn, mua, mus, g, n, f,...
      phase0, n_photons, disable_pbar, uint64(rnseed), false, use_alt);
    t_end_double = toc(t_start);
    double_res(use_alt + 1) = t_end_double;

    fprintf('%s CPU single precision version\n', text);
    t_start = tic;
    [element_fluence, boundary_exitance, boundary_fluence, simulation_time, seed_used] = MC3Dmex(...
      H, HN, BH, single(r), BCType, single(BCIntensity), BCLightDirectionType,...
      single(BCLightDirection), single(BCn), single(mua), single(mus), single(g), single(n), single(f),...
      single(phase0), n_photons, disable_pbar, uint64(rnseed), false, use_alt);
    t_end_single = toc(t_start);
    single_res(use_alt + 1) = t_end_single;

  end

  fprintf('GPU double precision version\n');
  t_start = tic;
  [element_fluence, boundary_exitance, boundary_fluence, simulation_time, seed_used] = MC3Dmex(...
    H, HN, BH, r, BCType, BCIntensity, BCLightDirectionType,...
    BCLightDirection, BCn, mua, mus, g, n, f,...
    phase0, n_photons, disable_pbar, uint64(rnseed), true, false);
  t_end_double_gpu = toc(t_start);

  fprintf('GPU single precision version\n');
  t_start = tic;
  [element_fluence, boundary_exitance, boundary_fluence, simulation_time, seed_used] = MC3Dmex(...
    H, HN, BH, single(r), BCType, single(BCIntensity), BCLightDirectionType,...
    single(BCLightDirection), single(BCn), single(mua), single(mus), single(g), single(n), single(f),...
    single(phase0), n_photons, disable_pbar, uint64(rnseed), true, false);
  t_end_single_gpu = toc(t_start);

  fprintf('GPU single precision version took %f sec\n', t_end_single_gpu);
  fprintf('GPU double precision version took %f sec\n', t_end_double_gpu);
  fprintf('GPU single precision version %f times faster\n', speedup(t_end_double_gpu, t_end_single_gpu));

  fprintf('Reference CPU single precision version took %f sec\n', single_res(1));
  fprintf('Alternate CPU single precision version took %f sec\n', single_res(2));
  fprintf('Alternate CPU single precision version %f times faster\n', speedup(single_res(1), single_res(2)));

  fprintf('Reference CPU double precision version took %f sec\n', double_res(1));
  fprintf('Alternate CPU double precision version took %f sec\n', double_res(2));
  fprintf('Alternate CPU double precision version %f times faster\n', speedup(double_res(1), double_res(2)));

  fprintf('Reference CPU single precision version %f times faster\n', speedup(double_res(1), single_res(1)));
  fprintf('Alternate CPU single precision version %f times faster\n', speedup(double_res(2), single_res(2)));
end


function res = speedup (slower, faster)
  res = 1.0 / (1.0 - faster/slower)
end
