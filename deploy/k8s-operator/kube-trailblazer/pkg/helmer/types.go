package helmer

import (
	helmclient "github.com/mittwald/go-helm-client"
	"github.com/nvidia/kube-trailblazer/pkg/clients"
	"helm.sh/helm/v3/pkg/chartutil"
)

// Type Guard asserting that Helmer satisfies the Helmer interface.
var _ Interface = &Helmer{}

// Helmer describes the resource to be built
type Helmer struct {
	Package    HelmPackage                   `json:"helmArbor"`
	Client     helmclient.Client             `json:"helmClient"`
	Options    helmclient.GenericHelmOptions `json:"helmOptions"`
	KubeClient clients.ClientsInterface      `json:"kubeClient"`
	Debug      bool                          `json:"debug"`
}

// Entry represents a collection of parameters for chart repository, since
// we cannot annotate the internal helm struct we're doing it here
type repoEntry struct {
	Name string `json:"name"`
	URL  string `json:"url"`
	// +kubebuilder:validation:Optional
	Username string `json:"username"`
	// +kubebuilder:validation:Optional
	Password string `json:"password"`
	// +kubebuilder:validation:Optional
	CertFile string `json:"certFile"`
	// +kubebuilder:validation:Optional
	KeyFile string `json:"keyFile"`
	// +kubebuilder:validation:Optional
	CAFile string `json:"caFile"`
	// +kubebuilder:validation:Optional
	InsecureSkipTLSverify bool `json:"insecure_skip_tls_verify"`
	// +kubebuilder:validation:Optional
	PassCredentialsAll bool `json:"pass_credentials_all"`
}

// A shelter of vines or branches or of latticework covered with climbing
// shrubs or vines, also latin for tree
type HelmPackage struct {
	RepoEntry repoEntry            `json:"repoEntry"`
	ChartSpec helmclient.ChartSpec `json:"chartSpec"`
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Schemaless
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	// TODO ChartValues json.RawMessage `json:"chartValues"`
	ChartValues chartutil.Values `json:"chartValues"`
	// +kubebuilder:validation:Optional
	ReleaseName string `json:"releaseName"`
}

type Pipeline []HelmPackage
