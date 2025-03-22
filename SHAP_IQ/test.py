import shapiq
from sklearn.ensemble import RandomForestRegressor

def explain_with_shaply_interactions():
    # load data
    X, y = shapiq.load_california_housing(to_numpy=True)
    # train a model
    model = RandomForestRegressor()
    model.fit(X, y)
    # set up an explainer with k-SII interaction values up to order 4
    explainer = shapiq.TabularExplainer(
        model=model,
        data=X,
        index="k-SII",
        max_order=4
    )
    print(X[0])
    print(X[0].shape)
    # explain the model's prediction for the first sample
    interaction_values = explainer.explain(X[0], budget=256)
    # analyse interaction values
    print(interaction_values)

    # >> InteractionValues(
    # >>     index=k-SII, max_order=4, min_order=0, estimated=False,
    # >>     estimation_budget=256, n_players=8, baseline_value=2.07282292,
    # >>     Top 10 interactions:
    # >>         (0,): 1.696969079  # attribution of feature 0
    # >>         (0, 5): 0.4847876
    # >>         (0, 1): 0.4494288  # interaction between features 0 & 1
    # >>         (0, 6): 0.4477677
    # >>         (1, 5): 0.3750034
    # >>         (4, 5): 0.3468325
    # >>         (0, 3, 6): -0.320  # interaction between features 0 & 3 & 6
    # >>         (2, 3, 6): -0.329
    # >>         (0, 1, 5): -0.363
    # >>         (6,): -0.56358890
    # >> )

def compute_shaply_values():
    data, y = shapiq.load_california_housing(to_numpy=True)
    # data = shapiq.load_california_housing(to_numpy=True)
    # train a model
    model = RandomForestRegressor()
    model.fit(data, y)
    # data, model = ...  # get your data and model
    explainer = shapiq.TreeExplainer(
        model=model,
        # data=data,
        index="SV",  # Shapley values
    )
    print(type(data))
    print(data[0])
    print(data[0].shape)
    print(type(data[0]))
    shapley_values = explainer.explain(data[0])
    shapley_values.plot_force()
    # shapley_values.plot_force(feature_names=...)
    
    # You can add interaction here like:
    # explainer = shapiq.Explainer(
    #     model=model,
    #     data=data,
    #     index="k-SII",  # k-SII interaction values
    #     max_order=2     # specify any order you want
    # )
    # interaction_values = explainer.explain(data[0])
    # interaction_values.plot_force(feature_names=...)
    
compute_shaply_values()