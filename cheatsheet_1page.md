<div style="font-family:Arial,sans-serif;font-size:8px;line-height:1.05;color:#111;">

<div style="page-break-after:always;break-after:page;column-count:3;column-gap:10px;">

<h1 style="font-size:11px;margin:0 0 3px;">ML Final Part A Cheatsheet - Page 1</h1>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">W8 Trees / Ensembles</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Model</th><th>Idea</th><th>Output</th></tr>
<tr><td>Class tree</td><td>Split X; leaf=majority class.</td><td><code>DecisionTreeClassifier(max_depth=d)</code>; score=accuracy</td></tr>
<tr><td>Reg tree</td><td>Leaf=mean y.</td><td>MSE low good; RMSE=sqrt(MSE)</td></tr>
<tr><td>Prune</td><td>Limit complexity.</td><td>max_depth, min_samples_leaf; deep=overfit</td></tr>
<tr><td>Bagging</td><td>Bootstrap many trees, all predictors.</td><td><code>RandomForestRegressor(max_features=p)</code></td></tr>
<tr><td>RF</td><td>Bagging + random predictors.</td><td><code>feature_importances_</code>; lowers variance</td></tr>
<tr><td>Boosting</td><td>Sequential trees learn errors.</td><td>accurate; high n_estimators/lr may overfit</td></tr></table>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">tree=DecisionTreeClassifier(max_depth=6).fit(Xtr,ytr)
pred=tree.predict(Xte); confusion_matrix(yte,pred); classification_report(yte,pred)
rf=RandomForestRegressor(max_features=4).fit(Xtr,ytr)
mean_squared_error(yte,rf.predict(Xte)); rf.feature_importances_</pre>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">W9 SVM</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Term</th><th>Meaning</th></tr>
<tr><td>Hyperplane</td><td>Decision boundary; linear SVM=straight.</td></tr>
<tr><td>Margin</td><td>Distance to closest points; larger often better.</td></tr>
<tr><td>Support vectors</td><td>Closest points determining boundary.</td></tr>
<tr><td>C</td><td>Small C=wider margin, more train errors. Large C=fewer train errors, overfit risk.</td></tr>
<tr><td>gamma</td><td>RBF locality; large gamma=wiggly/local.</td></tr>
<tr><td>Kernel</td><td>linear/rbf/poly; RBF for nonlinear.</td></tr></table>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">svc=SVC(C=1,kernel="linear").fit(Xtr,ytr); svc.score(Xte,yte)
params=[{"C":[.01,.1,1,10,100],"gamma":[.5,1,2,3,4]}]
clf=GridSearchCV(SVC(kernel="rbf"),params,cv=10,scoring="accuracy").fit(Xtr,ytr)
clf.best_params_; clf.best_score_</pre>
<p style="margin:1px 0;">ROC: top-left good. AUC 1 perfect, 0.5 random. Train high + test low = overfit.</p>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">W10 PCA</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Term</th><th>Meaning</th></tr>
<tr><td>Unsupervised</td><td>Only X, no y; find structure.</td></tr>
<tr><td>PCA</td><td>PCs=linear combinations of original variables.</td></tr>
<tr><td>PC1/PC2</td><td>PC1 largest variance; PC2 next; PCs uncorrelated.</td></tr>
<tr><td>Scores</td><td>Observation coordinates; plot observations.</td></tr>
<tr><td>Loadings</td><td>Variable weights; large abs=influential; sign=direction.</td></tr>
<tr><td>PVE/Scree</td><td>Variance explained; cumulative PVE total captured; elbow useful.</td></tr></table>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">Xsc=pd.DataFrame(scale(df),index=df.index,columns=df.columns)
pca=PCA(); scores=pca.fit_transform(Xsc)
loadings=pd.DataFrame(pca.components_.T,index=df.columns)
pve=pca.explained_variance_ratio_; np.cumsum(pve)</pre>
<p style="margin:1px 0;">Scale before PCA if variables have different units.</p>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">W11 Clustering</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Method</th><th>Data</th><th>Center</th><th>Output</th></tr>
<tr><td>K-means</td><td>numeric</td><td>mean</td><td>labels_, centers_, inertia_</td></tr>
<tr><td>K-modes</td><td>categorical</td><td>mode</td><td>labels_, centroids_, cost_</td></tr>
<tr><td>K-medoids</td><td>numeric/dissim</td><td>actual point</td><td>robust outliers</td></tr>
<tr><td>K-prototypes</td><td>mixed</td><td>mean+mode</td><td>categorical=[cols]</td></tr></table>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">km=KMeans(n_clusters=3,n_init=20,random_state=123).fit(X)
km.labels_; km.cluster_centers_; km.inertia_ # lower better
md=KModes(n_clusters=4,init="Huang",n_init=5).fit(data)
md.labels_; md.cluster_centroids_; md.cost_ # lower better
kp=KPrototypes(n_clusters=3,init="Cao"); kp.fit_predict(df,categorical=[0,1])</pre>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">Manual: Euclidean=sqrt(sum squared diffs); Manhattan=sum abs diffs.
K-means: centroid=column means -> nearest -> update -> stop if unchanged.
K-modes: same=0,diff=1,sum -> lowest -> update mode -> state ties.
Example O(A,L,M),C(A,L,C): 0+0+1=1.</pre>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">W12 Hierarchical / EM</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Item</th><th>Meaning</th></tr>
<tr><td>Hierarchical</td><td>Dendrogram; no K first.</td></tr>
<tr><td>Agglomerative</td><td>Bottom-up; each obs starts alone.</td></tr>
<tr><td>Height</td><td>Merge dissimilarity; lower=similar; vertical axis matters.</td></tr>
<tr><td>Complete</td><td>farthest pair; compact.</td></tr>
<tr><td>Single</td><td>closest pair; chaining.</td></tr>
<tr><td>Average</td><td>average pairwise distance.</td></tr>
<tr><td>EM/GMM</td><td>soft clustering; E probs, M updates.</td></tr></table>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">hc=linkage(X,method="complete",metric="euclidean")
dendrogram(hc); labels=cut_tree(hc,n_clusters=4).reshape(-1)
gmm=GaussianMixture(n_components=2).fit(X); gmm.predict(X); gmm.predict_proba(X)</pre>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">W13 Neural Nets</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Term</th><th>Meaning</th></tr>
<tr><td>Perceptron</td><td>one neuron; simple linear boundary.</td></tr>
<tr><td>MLP</td><td>hidden layers; nonlinear.</td></tr>
<tr><td>Feed-forward</td><td>input -> weights/bias -> activation -> output.</td></tr>
<tr><td>Backprop</td><td>updates weights from error.</td></tr>
<tr><td>Activation</td><td>ReLU/tanh/sigmoid nonlinearity.</td></tr>
<tr><td>Scaling</td><td>essential. hidden_layer_sizes=(10,10,10).</td></tr></table>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000).fit(Xtr,ytr.values.ravel())
confusion_matrix(yte,mlp.predict(Xte)); classification_report(yte,mlp.predict(Xte))
reg=MLPRegressor(activation="relu",hidden_layer_sizes=(16,)).fit(Xtr,ytr)
mean_squared_error(yte,reg.predict(Xte)); r2_score(yte,reg.predict(Xte))</pre>

</div>

<div style="column-count:3;column-gap:10px;">

<h1 style="font-size:11px;margin:0 0 3px;">ML Final Part A Cheatsheet - Page 2</h1>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">Output Interpretation</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Output</th><th>Meaning</th><th>Say</th></tr>
<tr><td>.score classifier</td><td>accuracy</td><td>higher test better</td></tr>
<tr><td>.score regressor</td><td>R2</td><td>1 perfect, 0 mean, negative bad</td></tr>
<tr><td>confusion matrix</td><td>counts</td><td>diagonal correct</td></tr>
<tr><td>precision</td><td>TP/(TP+FP)</td><td>low=many false positives</td></tr>
<tr><td>recall</td><td>TP/(TP+FN)</td><td>low=many false negatives</td></tr>
<tr><td>F1</td><td>balance</td><td>good for imbalance</td></tr>
<tr><td>support</td><td>true count</td><td>check imbalance</td></tr>
<tr><td>MSE/RMSE</td><td>error</td><td>lower better; RMSE y units</td></tr>
<tr><td>coef</td><td>effect</td><td>sign=direction</td></tr>
<tr><td>P>|t|</td><td>p-value</td><td><.05 significant</td></tr>
<tr><td>R2/adj R2</td><td>fit</td><td>adj better compare p</td></tr>
<tr><td>AIC/BIC</td><td>criteria</td><td>lower better; BIC harsher</td></tr>
<tr><td>CV</td><td>test estimate</td><td>choose best CV</td></tr></table>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">Confusion matrix:
                 Pred0 Pred1
Actual0           TN    FP
Actual1           FN    TP
Acc=(TP+TN)/N; Precision=TP/(TP+FP); Recall=TP/(TP+FN)</pre>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">Core Formulas</h2>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">Linear: y=b0+b1x1+...+bpxp+e
Residual=actual-pred; RSS=sum(resid^2); MSE=mean(resid^2); RMSE=sqrt(MSE)
Logistic: p=exp(eta)/(1+exp(eta)), eta=b0+b1x; odds=p/(1-p); log-odds=eta
Euclidean=sqrt((a1-b1)^2+(a2-b2)^2+...); Manhattan=|a1-b1|+|a2-b2|+...
PVE=PC variance/total variance; K-means inertia=sum squared distance to own centroid</pre>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">Week 1-7 Safety Net</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Week</th><th>High-yield idea</th><th>Command</th></tr>
<tr><td>W1</td><td>Reg quantitative y; class categorical y; clustering/PCA no y. Test estimates future.</td><td>train_test_split, LinearRegression, KMeans, PCA</td></tr>
<tr><td>W2</td><td>Coef sign; p-value relation; residuals/outliers/leverage. Polynomial still linear in beta.</td><td>smf.ols(...).summary()</td></tr>
<tr><td>W3</td><td>Logistic probability -> threshold -> class. LDA Gaussian/common covariance.</td><td>LogisticRegression, LDA, confusion_matrix</td></tr>
<tr><td>W4</td><td>Validation/LOOCV/k-fold estimate test error. Bootstrap replacement for uncertainty.</td><td>cross_val_score, LeaveOneOut, resample</td></tr>
<tr><td>W5</td><td>Best subset all; forward/backward search. Ridge shrinks; lasso zeros variables.</td><td>RidgeCV, LassoCV; lower CV/AIC/BIC</td></tr>
<tr><td>W6</td><td>PCR=PCA then regression, choose PCs by CV. PLS supervised uses y.</td><td>PCA, LinearRegression, CV MSE</td></tr>
<tr><td>W7</td><td>Polynomial, step, splines, local reg, GAM. Choose complexity by p-values/CV.</td><td>PolynomialFeatures, pd.cut, dmatrix("bs"), anova_lm</td></tr></table>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">Common Commands</h2>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,mean_squared_error,r2_score
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=0)
res=smf.ols("medv~lstat+rm",data=df).fit(); res.summary()
lr=LinearRegression().fit(Xtr,ytr); lr.coef_; lr.intercept_
log=LogisticRegression(C=100000,tol=1e-7).fit(Xtr,ytr)
scores=cross_val_score(model,X,y,cv=10,scoring="accuracy")
errs=-cross_val_score(model,X,y,cv=10,scoring="neg_mean_squared_error")
ridge=RidgeCV(alphas=np.logspace(-3,3,100)).fit(Xtr,ytr)
lasso=LassoCV(alphas=np.logspace(-3,3,100),max_iter=100000).fit(Xtr,ytr)
lasso.coef_ # zeros=removed
X4=PolynomialFeatures(degree=4).fit_transform(X)
spl=dmatrix("bs(age,df=6,include_intercept=False)",{"age":df.age},return_type="dataframe")</pre>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">Fast Comparisons</h2>
<table style="width:100%;border-collapse:collapse;font-size:7px;"><tr><th>Compare</th><th>Answer</th></tr>
<tr><td>train vs test</td><td>train optimistic; test unseen performance</td></tr>
<tr><td>bias/variance</td><td>simple high bias low var; flexible low bias high var</td></tr>
<tr><td>bagging/RF</td><td>bagging all predictors; RF random subset</td></tr>
<tr><td>RF/boosting</td><td>independent vs sequential trees</td></tr>
<tr><td>PCA/clustering</td><td>reduce/visualize vs group observations</td></tr>
<tr><td>K-means/hier</td><td>need K first vs dendrogram then cut</td></tr>
<tr><td>K-means/K-modes</td><td>numeric mean vs categorical mode</td></tr>
<tr><td>hard/soft</td><td>one label vs membership probabilities</td></tr>
<tr><td>ridge/lasso</td><td>shrink vs shrink + select</td></tr>
<tr><td>PCR/PLS</td><td>PCR ignores y for PCs; PLS uses y</td></tr></table>

<h2 style="font-size:9px;margin:3px 0 1px;border-bottom:1px solid #999;">Answer Frames</h2>
<pre style="font-size:6.7px;line-height:1.02;margin:1px 0;padding:2px;background:#f5f5f5;white-space:pre-wrap;">Output: Test ___=___. Means ___. Better/worse than ___ because ___. Train>>test => overfit.
Select model: lower test/CV MSE or higher test/CV accuracy/F1; lower AIC/BIC.
K-means: centroid -> distance -> nearest -> update -> stop unchanged.
K-modes: same0/diff1 -> sum -> lowest -> update mode -> mention ties.</pre>

</div>

</div>
