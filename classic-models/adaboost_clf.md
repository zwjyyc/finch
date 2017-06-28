fit
```
for i = 1 : n_models
    models[i].train(X, y, sample_weight=w)      # train a weighted classifier
    pred = models[i].predict(X)                 # compute predictions
    err = dot(w , y != pred)                    # compute weighted error rate
    alphas[i] = 0.5 * log( (1 - err) / err )    # compute coefficient
    w *= exp(- alphas[i] * y * pred )           # update weights
    w /= sum(w)
end
```
predict
```
sign( add( alphas[i] * models[i].predict(X_test) ) )
```
