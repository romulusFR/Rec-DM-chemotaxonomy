# SOme backups

```python
# TODO : les couleurs par classes

ca = prince.CA(n_components=2, n_iter=5, copy=True, check_input=True, engine="auto", random_state=42, benzecri=False)

confused_df = pd.DataFrame(100 * confused)
confused_df.index = with_with_matrix.index  #  set([(x,y) for x,y,_ in dataset.index.to_list()])
confused_df.columns = with_with_matrix.columns
# confused_df.values = confused
# confused_df


/!\ HERE /!\

for (df, name) in confused: # [(with_with_matrix, "Original"), (confused_df, "Confused")]:
    print(f"-------{name}-------")
    ca = ca.fit(df)

    # ca.row_coordinates(df)[:10]
    # ca.column_coordinates(df)[:10]

    pprint(ca.explained_inertia_)
    pprint(ca.col_masses_[:10])
    pprint(ca.eigenvalues_)
    pprint(ca.total_inertia_)

    ax = ca.plot_coordinates(
        X=confused_df,
        ax=None,
        figsize=(12, 12),
        x_component=0,
        y_component=1,
        show_row_labels=True,
        show_col_labels=True,
    )
    plt.show()
```
