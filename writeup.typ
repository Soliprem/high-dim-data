#import "@preview/physica:0.9.6": *
#import "@preview/beautiframe:0.1.0": *
#import "@preview/equate:0.3.2": *
#import "@preview/cetz:0.4.2": *
#beautiframe-setup(style: "minimal", link-to-section: true)
#show: equate.with(breakable: true, sub-numbering: true)
#set page(numbering: "1")
#set math.equation(numbering: "(1.1)")
#let remark(body) = block(
  width: 100%,
  inset: (left: 8pt, top: 4pt, bottom: 4pt),
  stroke: (left: 2pt + gray),
)[
  #body
]

#let authors = (
  (
    name: "Francesco Prem Solidoro",
    email: "francesco.solidoro@studio.unibo.it",
    matricola: "0001233485"
  ),
  (
    name: "Michele Salvi",
    email: "michele.salvi4@studio.unibo.it",
    matricola: "0001234153"
  ),
)

#let title = [Clustering for Exploratory Factor Analysis]

#set align(center)
#text(size: 17pt, weight: "bold")[#title]
#v(0.5em)

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  row-gutter: 1em,
  ..authors.map(author => [
    #text(size: 11pt, weight: "semibold")[#author.name] \
    #text(size: 9pt, style: "italic")[#author.email] \
    #text(size: 9pt)[Matricola: #author.matricola]
  ])
)

#v(1em)
#set align(left)
#set heading(numbering: "1.")

#outline(depth: 2)


= Introduction
//When dealing with data of high dimensionality multiple dimensionality reduction techniques are available, each with its own strengths and weaknesses.

High-dimensional datasets often contain many correlated variables, making direct interpretation difficult. Dimensionality-reduction techniques aim to summarize this information using a smaller number of informative dimensions. In the case of economic and development indicators, many observed variables are likely to reflect common latent structures, such as trade openness, institutional capacity, service-sector development, or financial exposure. This makes exploratory factor analysis a natural modelling tool.

However, country-level development data may also be heterogeneous. Relationships among indicators may differ across groups of economies. A single global factor model may therefore be too restrictive if it assumes one common covariance structure for all countries.

== Research Question
The objective of this paper is to determine whether conditioning on service productivity and institutional-logistics capacity reveals heterogeneous covariance regimes, allowing cluster-specific factor models to achieve a lower conditional regularized BIC than a single global factor model. The underlying hypothesis is that countries operating at different levels of service productivity and logistics capacity may exhibit different relationships among trade, investment, debt, inflation, services, and institutional indicators.

= Data

The analysis uses the WDI 2022 dataset provided for the course. The dataset contains country-level economic, trade, investment, service-sector, institutional, fiscal, and price indicators for 217 economies. Identification variables, including country name, ISO3 code, World Bank region, income group, and year, were retained only for interpretation and were excluded from clustering and factor analysis.

After preprocessing and comparability filtering, the final factor-analysis dataset contains 43 numerical indicators. These variables are suitable for exploratory factor analysis because many WDI indicators are correlated and may reflect common latent economic dimensions, such as trade openness, service-sector development, investment structure, debt exposure, institutional capacity, and inflation dynamics.

The empirical objective is to test whether estimating factor models separately within clusters of countries improves the representation of this covariance structure relative to a single global factor model.

= Theoretical Overview
The following section will go over the theory underlying the analysis methods that will be used in the paper.
== Clustering
Cluster Analysis partitions observations in groups called "clusters", with the goal of determining optimal cluster allocation and the number of clusters itself.

We will be using k-means clustering in the paper, which requires the number of clusters to be pre-specified, in a later section we will expand on the specifics.

WDI actually contains some Labelled data, however the available classes are on income while for our analysis we preferred to cluster on other variables, to proxy productivity.

All types of cluster analysis generally follow two steps:
+ Measuring the distance between object pairs
+ Choosing the method of clustering, that means the ruleset for the formation of clusters based on distances#footnote[we will be talking about distance and not similarity as our dataset contains basically no categorical variables]

Distances are measured via the distance matrix, which contains distances between objects, where the distance between object $i$ and object $j$ is located in the $i$-th row and $j$-th column. \
The distance matrix is obtained by choosing a distance type, for which there are different alternatives:
- Euclidean distance: $ quad delta_(i j) = sqrt(sum_(k = 1)^d (x_(i k) - x_(j k))^2) $
- Manhattan distance: $ quad delta_(i j) = sum_(k = 1)^d abs(x_(i k) - x_(j k)) $

Euclidean distance emphasizes large differences because deviations are squared, whereas Manhattan distance treats deviations linearly and is therefore less sensitive to extreme observations. \


Clustering methods instead can be either hierarchical or non-hierarchical. \

Hierarchical methods lead to a structure in which subsets of clusters of one level are aggregated to form clusters at the next, which can be further divided into:
- *agglomerative*: each object is treated as a one-member cluster and then clustered together progressively#footnote[agglomerative since once a pair is formed they can never be separated]
- *divisive*: the whole set of observations is treated as a single cluster and then split up into clusters
Some notable hierarchical methods are the nearest neighbor / single linkage or farthest neighbor / complete linkage.

Non-hierarchical methods instead move individuals in and out of clusters. They assume k clusters a priori, after which there are different methods to divide observations among the clusters.\
One of the most common, and the one we used in this paper, is the k-means clustering, which partitions observations into K groups by minimizing within-cluster sum of squared Euclidean distances. Each observation is assigned to the nearest cluster centroid.

In our specific case, due to our usage of k-means we can avoid using pairwise distance measures. Hierarchical methods explicitly use a distance matrix, while k-means iteratively assigns observations to centroids based on Euclidean distance.
== Factor Analysis
=== General Model
Factor analysis is an analysis method that highlights the dependence of observed variables and reduces dimensionality by expressing them in a smaller set of latent variables #footnote[variables that can't be measured directly]. \
Since we expect indicators of the same latent variable to be correlated we can analyze the correlation matrix and construct a factor model where, given correlated $X_1, X_2, ...X_d$ variables that depend on the common factor $f_i$. \
Factor loadings $lambda_(i j)$ are estimated from the covariance structure of the data and describe the relationship between observed variables and latent factors. Under the factor model, the covariance matrix can be expressed as: $bold(Sigma) = bold(Lambda) bold(Lambda)' + bold(Psi)$  \
Then, the general model is obtained as:
$
X_i = mu_i + lambda_( i 1) f_1 + lambda_( i 2) f_2 + ...lambda_( i q) f_q + u_i \
bold(X) = mu + bold(Lambda) F + u 
$
Where the optimal number of factors are obtained such that the latent structure adequately reproduces the observed covariance matrix. \
In the model:
- $bold(X)$ : observed variables
- $mu$: vector containing the means of $bold(X)$
- F: vector of factors
- u: vector of errors 
- $bold(Lambda)$: Matrix of factor loadings

The model additionally relies on the following assumptions:
+ Factors have null mean, unit variance and are uncorrelated:\ $ quad E(f) = 0 quad E(f f') = I$
+ Errors have null mean, are uncorrelated with each other and with factors and have variance $Psi$:\ $quad E(u) = 0 quad E(u u') = Psi quad E(f u') = 0$ 
Additionally, since we are conducting Exploratory Factor Analysis (EFA) we are assuming that observed variables are linearly related to the latent factors.

Regarding the Variance and Variability of the data, we make 2 additional considerations:

The variance of $x_i$ (after standardization) is: $ V(x_i) = sum_(j = 1)^q lambda_(i j)^2 + Psi_(i i) $ 
where the first term, _communality_ represents variance explained by the latent factors, and the second term _uniqueness_ represents the leftover variance not explained by the factors

The data's variability can also be expressed in matrix form by considering $Sigma$, the covariance matrix of $x$, of dimension $d * d$ as it holds that: $ bold(Sigma) = bold(Lambda) bold(Lambda)' + bold(Psi) $
where:
- $bold(Lambda) bold(Lambda)'$ is the variance-covariance matrix of factor loadings
- $bold(Psi)$ is a diagonal matrix of individual $Psi_(i i)$

// Since we will estimate the factor model, we need to ensure our model is identified. \
// Factor models are generally not uniquely identified because different loading matrices can generate the same covariance structure through orthogonal transformations. Identification constraints are therefore required, typically by fixing factor variances and imposing rotational restrictions.


// CJ
Since we estimate the factor model, we must ensure that it is identified.

For a factor model with $d$ observed variables and $q$ factors, the covariance
structure contains $d q$ loading parameters and $d$ uniqueness parameters.
Because the model is rotationally invariant, $q(q-1)/2$ restrictions are
required. The number of covariance parameters is therefore

$
p_"cov" = d q + d - (q(q-1))/2.
$

When variable means are included in the likelihood, the total parameter count
used for AIC and BIC is

$
p = d q + 2d - (q(q-1))/2.
$

For identification, the number of freely estimated covariance parameters must
not exceed the $d(d+1)/2$ distinct elements of the observed covariance matrix.
In practice, a substantially overidentified model is preferable because the
objective is to reproduce the covariance structure using considerably fewer
latent dimensions.
// CJ


=== Estimation
Variables were standardized once over the full sample. The global factor model therefore uses a correlation matrix, whereas the cluster-specific models use covariance matrices expressed on the same globally standardized scale. This preserves comparability across clusters, but cluster-specific loadings are not necessarily correlations and can occasionally exceed one in absolute value.

Estimation follows iteratively the following steps:
+ $bold(Sigma)$ is estimated by the observed covariance matrix, $bold(S)$
+ Given $bold(S)$ we need to estimate $bold(hat(Sigma)) "and" bold(hat(Psi))$ so that $bold(S)$ approximates $bold(Sigma)$ as closely as possible, i.e. $bold(S) approx bold(hat(Lambda)) bold(hat(Lambda))' + bold(hat(Psi))$
We estimate the needed matrices via maximum likelihood, whose likelihood-based interpretation relies on multivariate normality. #footnote[The matrices could alternatively be estimated through least squares or principal-factor methods, but this paper uses maximum likelihood.] \
We must also choose the appropriate number of factors.

Maximum likelihood estimation uses iterative numerical optimization procedures to obtain estimates of the loading matrix and uniqueness parameters.

=== Rotation and Factor Scores
Factor models are invariant to orthogonal rotations on loadings. Rotation aims to obtain a simple structure, where each variable loads strongly on a small number of factors and weakly on others.

Factor Loadings are easily interpreted by ensuring that each variable is loaded mostly on one factor and that all loadings are either large or close to zero.


Additionally, we can choose from two types of rotations:
+ Orthogonal rotations: factors are restricted to be uncorrelated. When variables are standardized within the modeled sample, loadings can be interpreted as variable--factor correlations
+ Oblique rotations: allow for factor correlation, which is more realistic and may lead to a better model fit

In the paper we use Varimax, a kind of orthogonal rotation that constrains factors to have few large loadings while most take near zero values.

Once an appropriate rotation is found all that's left is estimating the factor scores, which can be done via different methods. Some example methods are:
- Bartlett's method, which uses MLE, and scores for the ith individual are:
  $ hat(f_i) = (bold(hat(Lambda))' bold(hat(Psi))^(-1) bold(hat(Lambda)))^(-1) bold(hat(Lambda))' bold(hat(Psi))^(-1) x_i  $
- Thomson's method, where estimates of factor scores are obtained as in a Bayesian framework. Scores for the ith individual are:
  $ hat(f_i) = ( I + bold(hat(Lambda))' bold(hat(Psi))^(-1) bold(hat(Lambda)) )^(-1) bold(hat(Lambda))' bold(hat(Psi))^(-1) x_i $

Factor scores were estimated using the regression-based scoring procedure implemented in scikit-learn.

== Trade-off
Estimating separate factor structures within clusters may better capture heterogeneous covariance patterns, but it reduces the effective sample size available for each model and increases model complexity. Excessive partitioning may therefore lead to unstable estimates and overfitting. Furthermore, clustering introduces dependence on the selected partitioning scheme, since different clustering specifications may produce different covariance structures and therefore different factor solutions.


= Methods
The empirical strategy follows four main steps: data preprocessing, clustering, factor analysis, and model comparison.

All of the code used for this project can be found in #link("https://github.com/soliprem/high-dim-data")[this repository.]

== Data Preprocessing

We analyzed the WDI 2022 dataset. Identification variables, including country
name, ISO3 code, region, income group, and year, were retained for
interpretation but excluded from clustering and factor analysis.

The original WDI indicator codes were replaced with descriptive names.
Eleven variables expressed in current or constant local-currency units were
removed because their numerical scales are not comparable across countries
using different national currencies.

Current-USD measures of FDI inflows, FDI outflows, net FDI, and the trade
balance were replaced by GDP-normalized alternatives. Existing percentage-of-
GDP indicators were used where available, while net FDI as a percentage of GDP
was derived from the available national-accounts variables.

Variables were converted to numeric format and missing observations were
imputed using the feature median. Extreme observations were winsorized at the
1st and 99th percentiles. A Yeo--Johnson power transformation was then applied
feature by feature. Unlike the logarithmic transformation, Yeo--Johnson can
also accommodate zero and negative values. Finally, all variables were
standardized to zero mean and unit variance.

After preprocessing, only two of the 43 factor variables had an absolute
skewness greater than one. Seven variables nevertheless retained excess
kurtosis greater than three, indicating that some heavy-tailed behavior
remained, particularly among investment-flow indicators.

#table(
  columns: 3,
  [*Step*], [*Action*], [*Reason*],

  [Variable renaming], [WDI codes renamed], [Interpretability],
  [Local-currency variables], [Removed], [Not comparable across countries],
  [USD FDI/trade balance flows], [Removed or replaced by % GDP], [Avoid country-size effects],
  [Net FDI], [Derived as % GDP], [Retain FDI balance in comparable form],
  [Missing values], [Median imputation], [Avoid listwise deletion],
  [Extreme values], [1st/99th percentile winsorization], [Limit outlier influence],
  [Skewness], [Yeo-Johnson transformation], [Handle zero/negative values],
  [Scaling], [Standardization], [Prevent scale dominance and obtain a correlation-scale global analysis],
)

== Clustering Specifications

We used k-means clustering to divide countries into groups before estimating factor models. Since k-means requires the number of clusters to be chosen in advance, we tested several specifications. In particular, we considered 2 to 6 clusters and three different sets of clustering variables:
- services value added per worker;
- services value added per worker and logistics performance;
- services value added per worker, logistics performance and statistical performance.

Each k-means specification used 50 random initializations and a fixed random
seed to improve reproducibility and reduce sensitivity to poor centroid
initialization.

The aim of clustering was not to classify countries according to official categories, such as income groups, but to identify empirical regimes in which the relationship between development, services, trade, investment, and institutional indicators may differ.

For each specification, we computed the silhouette score as a diagnostic measure of cluster separation.

== Factor analysis specifications

After clustering, we estimated exploratory factor-analysis models (EFAs). We started with a global estimation using all countries together and then estimated separate factor models within each cluster. This allowed us to compare whether the covariance structure of the data is better represented by one global model or by cluster-specific models.

All factor models were estimated using maximum-likelihood factor analysis with Varimax rotation.

We evaluated factor models using several diagnostics:

- the average absolute primary loading;
- the average gap between primary and secondary loadings;
- the share of variables with a simple factor structure;
- the covariance reconstruction RMSE;
- the regularized log-likelihood;
- the regularized AIC and BIC.

The simple-structure share was defined as the proportion of variables whose
largest absolute loading was at least 0.40 and whose second-largest absolute
loading was at most 0.30.

The reconstruction RMSE measures how closely the model-implied covariance matrix approximates the empirical covariance matrix. However, RMSE tends to improve as more factors are added. Therefore, it was mostly used as a diagnostic measure rather than as the main selection criterion.

The main model-selection criterion was the regularized BIC. BIC penalizes model complexity and is therefore appropriate when comparing specifications with different numbers of factors. Since some high-dimensional factor models produced near-singular covariance matrices, an eigenvalue floor of $10^(-6)$ was used to compute a regularized likelihood. For this reason, the likelihood-based comparison should be interpreted as a regularized model-selection criterion rather than as an exact likelihood comparison.

In general, we define improvement as obtaining a more parsimonious representation of the covariance structure while maintaining interpretability and goodness of fit.
== Model comparison criteria

The central comparison of the project is between a global factor model and cluster-specific factor models. For each clustering specification and number of factors, we estimated one global factor model on the full dataset, and separate factor models within each cluster. We then compared whether the clustered specification improved the representation of the covariance structure of the WDI indicators.

The main selection criterion was the Bayesian Information Criterion (BIC). BIC is a penalized likelihood criterion: lower values indicate a better trade-off between model fit and model complexity. This is important because increasing the number of factors mechanically improves the likelihood and covariance reconstruction, but also increases the number of estimated parameters and reduces interpretability.

For global models, we computed BIC using the full-sample log-likelihood, the total number of observations, and the number of parameters of the global factor model.

For clustered models, we computed BIC as the sum of cluster-specific BIC values:

$
"BIC"_"clustered" = sum_(c=1)^k (p_c log(n_c) - 2 ell_c)
$

where ($p_c$) is the number of parameters estimated in cluster ($c$), ($n_c$) is the number of observations in cluster ($c$), and ($ell_c$) is the regularized log-likelihood of the factor model fitted within cluster ($c$). Cluster assignments were treated as fixed after k-means clustering.

This criterion is conditional on the clustering partition. It does not estimate a full joint likelihood for the clustering step and the factor-analysis step together. Therefore, the BIC comparison should be interpreted as comparing factor models conditional on the selected clusters.

We considered a fixed-factor model admissible if it successfully estimated all cluster-specific factor models, produced a finite regularized clustered BIC, had no cluster model with uniquenesses below the likelihood eigenvalue floor, and used no more factors than the smaller of the global PA cap and the smallest cluster size.

== Parallel analysis
Parallel analysis determines the number of factors by comparing the eigenvalues of the observed correlation matrix with eigenvalues obtained from permuted versions of the observed data with the same dimensions. Factors are retained only when their observed eigenvalues exceed the corresponding random eigenvalues.

The number of factors was assessed through permutation-based parallel analysis.
For each scope, the observed eigenvalues were compared with the 95th
percentile of eigenvalues obtained from 500 independently permuted datasets.
An observed factor was retained when its eigenvalue exceeded the corresponding
permutation threshold. This procedure suggested eight factors for the global
sample.

=== Adaptive parallel-analysis regime

Parallel analysis suggested eight factors for the global model. For the
three-cluster adaptive specification, it suggested respectively 5, 6, and 7
factors.

The adaptive aggregate obtained a regularized BIC of 15063.88. However, the
six-factor model in the second cluster contained three uniqueness estimates
below the numerical regularization threshold. Its exact and regularized
likelihoods consequently differed substantially. Therefore, we classified the adaptive result
as unstable in this specification, and did not select it as the preferred model, despite its lower regularized BIC.

= Interpretation

#table(
  columns: 5,
  [*Model*], [*Clusters*], [*Factors*], [*Regularized BIC*], [*Interpretation*],

  [*Best unconstrained fixed-factor*],
  [6], [21], [7407.29],
  [Lowest fixed-factor BIC, but five cluster models are unstable],

  [*Best admissible fixed-factor*],
  [2], [8], [15543.27],
  [Best stable specification],

  [*Adaptive PA model*],
  [3], [5/6/7], [15063.88],
  [Lower BIC, but one cluster is near-singular],

  [*Global PA model*],
  [1], [8], [17443.41],
  [Stable global benchmark],
)

The comparison shows that clustering improves the factor-analysis model relative to the global benchmark. The global PA model used eight factors and obtained a regularized BIC of 17443.41. The best stable clustered specification also used eight factors, but estimated separate factor models within two clusters. Its regularized BIC was 15543.27, giving an improvement of about 1900 BIC points relative to the global model.

Under the primary model-quality criterion, conditional regularized BIC, this
constitutes a substantial improvement. Secondary diagnostics did not improve
uniformly. The covariance-reconstruction RMSE was 0.0580 for the clustered
model and 0.0539 for the global model, while the share of variables satisfying
the simple-structure criterion was 0.445 and 0.581 respectively. The evidence
therefore supports an improvement in the penalized likelihood--complexity
trade-off, although the global model retains slightly better reconstruction
and simple-structure diagnostics.

The best unconstrained fixed-factor model achieved a much lower BIC of 7407.29, using six clusters and twenty-one factors. However, this model is not selected as the preferred specification because it does not satisfy the admissibility criteria: it is highly complex and five cluster-specific factor models are unstable. It is therefore useful only as a numerical benchmark showing how much fit can improve when model complexity is left essentially unrestricted.

The adaptive PA model used three clusters and selected 5, 6, and 7 factors, respectively. Its regularized BIC was 15063.88, which is lower than the best stable fixed-factor model. However, one of the cluster-specific models was near-singular, with uniqueness estimates below the numerical regularization threshold. For this reason, the adaptive model is treated as a sensitivity check rather than as the preferred final specification. It shows, however, strong promise.

The final preferred specification is therefore the best admissible fixed-factor model: two clusters and eight factors, with clustering based on service productivity and logistics performance. This model provides a substantial improvement over the global eight-factor model while avoiding the numerical instability observed in the adaptive specification and the excessive complexity of the unconstrained model.

== Interpretation of the preferred clusters

The preferred two-cluster partition is substantively interpretable:

#table(
  columns: 4,
  [*Cluster*], [*Countries*], [*Median service value added per worker*], [*Median logistics score*],
  [1], [159], [9949 USD], [2.6],
  [2], [58], [48812 USD], [3.7],
)

Cluster 1 is heterogeneous in income composition: it contains 26 low-income,
51 lower-middle-income, 47 upper-middle-income, and 34 high-income economies,
in addition to one unclassified economy. Cluster 2 contains 51 high-income and
7 upper-middle-income economies. The partition therefore separates a
high-productivity, high-logistics group from a broader lower-capacity group,
without simply reproducing the official income classification.

== Interpretation of the preferred factors

Because a separate factor model was estimated within each cluster, factor
numbers do not necessarily represent the same latent construct across
clusters. Factor signs are also arbitrary. The table therefore summarizes the
dominant loading patterns within each model rather than treating factors with
the same number as directly equivalent.

#table(
  columns: 3,
  [*Factor*], [*Cluster 1: dominant indicators*], [*Cluster 2: dominant indicators*],
  [1], [GDP-deflator and consumer-price inflation], [Scale of imports, exports, and services],
  [2], [Scale of imports, exports, and services], [Investment and debt indicators; weak structure],
  [3], [Current and constant trade levels], [GDP-deflator and consumer-price inflation],
  [4], [Public and publicly guaranteed debt service], [Tax structure, public expenditure, and inflation],
  [5], [Service-sector scale; weak structure], [Trade openness and trade in services],
  [6], [Net FDI and FDI inflows], [Financial-service composition and external balance],
  [7], [External balance and export intensity], [FDI outflows],
  [8], [Trade in services and service-sector intensity], [Portfolio investment and transport services],
)

Several differences are economically meaningful. In Cluster 1, net FDI and
FDI inflows form a distinct factor, while external balance and service-sector
intensity appear on separate dimensions. In Cluster 2, FDI outflows form the
clearest investment factor, whereas trade openness has a strong independent
factor. These results suggest that trade, services, and investment indicators
do not share the same covariance structure in lower- and higher-capacity
economies.

= Conclusion 

This project investigated whether clustering countries before estimating exploratory factor analysis improves the representation of a high-dimensional WDI 2022 dataset.

The analysis supports this hypothesis under the conditional regularized BIC criterion. After preprocessing the data to improve cross-country comparability, parallel analysis suggested an eight-factor global structure. The corresponding global factor model was stable, but the clustered version with two clusters and eight factors achieved a substantially lower conditional regularized BIC. This indicates that cluster-specific models provide a better penalized likelihood--complexity trade-off than a single global factor model, although the global model retains slightly better covariance-reconstruction and simple-structure diagnostics.

The final preferred model is the stable two-cluster, eight-factor specification. This model balances fit, interpretability, and numerical stability. The unconstrained six-cluster, twenty-one-factor model achieved the lowest BIC overall, but was too complex and unstable to be interpreted as the final result. The adaptive PA model achieved a lower BIC than the selected fixed-factor model, but one of its cluster-specific factor models was near-singular, so it was retained only as a sensitivity check.

Overall, the results suggest that WDI indicators do not share one homogeneous covariance structure across all countries. Clustering countries by service productivity and logistics performance before factor analysis identifies more homogeneous regimes and improves model quality as measured by conditional regularized BIC.

The analysis remains exploratory. K-means cluster assignments were treated as fixed when estimating the factor models, so the BIC comparison is conditional on the selected clustering partition. In addition, the smaller cluster contains only 58 observations for 43 variables, meaning that even the stable eight-factor solution is sample-constrained. Future work could test alternative clustering algorithms, oblique rotations, and the stability of the selected factor structure across multiple WDI years.
