np.random.seed(42)
point_mass = np.zeros(100)
gaussian_component = np.random.normal(5, 2, 150)
x = np.concatenate([point_mass, gaussian_component])

    # EM algorithm parameters
sigma_0 = 1.0  # Fixed standard deviation for the point mass
tau_squared = 2.0  # Initial guess for variance of the Gaussian component
pi = 0.5  # Initial guess for mixing proportion

    # Run the EM algorithm
em = EMAlgorithmPointMassTruncated(x, sigma_0, tau_squared, pi)
pi, mu, tau_squared = em.run()

    # Print results
    print(f"Estimated \u03c0: {pi}")
    print(f"Estimated \u03bc: {mu}")
    print(f"Estimated \u03c4^2: {tau_squared}")