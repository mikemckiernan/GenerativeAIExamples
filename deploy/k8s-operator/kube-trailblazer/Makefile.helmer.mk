
CSPLIT ?= csplit - --prefix="" --suppress-matched --suffix-format="%04d_trailblazer_manifest.yaml"  /---/ '{*}' 1>/dev/null


.PHONY: go-mod
go-mod: ## Runs go mod tidy/vendor to sync vendor directory with go.mod.
	go mod tidy
	go mod vendor

.PHONY: patch
patch:
	cp .patches/root.go vendor/helm.sh/helm/v3/pkg/chart/.


.PHONY: helm-chart

helm-chart: manifests kustomize ## Deploy controller to the K8s cluster specified in ~/.kube/config.
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	cd helm-charts/kube-trailblazer/templates && $(KUSTOMIZE) build ../../../config/default | $(CSPLIT)
## Remove namespace creation, helm can do that ...
	rm helm-charts/kube-trailblazer/templates/0000_trailblazer_manifest.yaml


helm-install: helm-chart
	helm install kube-trailblazer helm-charts/kube-trailblazer  --create-namespace --namespace kube-trailblazer-system

helm-upgrade: helm-chart
	helm upgrade kube-trailblazer helm-charts/kube-trailblazer --namespace kube-trailblazer-system 

helm-uninstall: 
	helm uninstall kube-trailblazer --namespace kube-trailblazer-system 

helm-package: helm-chart
	helm package helm-charts/kube-trailblazer -d helm-charts/
