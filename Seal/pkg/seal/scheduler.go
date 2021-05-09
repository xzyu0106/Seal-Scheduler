package seal

import (
	"context"
	"github.com/xzyu0106/Seal-Scheduler/pkg/seal/score"

	//"github.com/xzyu0106/Seal-Scheduler/rconfig"

	//"fmt"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	//"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	//"sort"
)

const (
	Name = "seal"
)

var (
	_ framework.ScorePlugin      = &seal{}
	_ framework.ScoreExtensions  = &seal{}
)

type Args struct {
	KubeConfig string `json:"kubeconfig,omitempty"`
	Master     string `json:"master,omitempty"`
}

type seal struct {
	args   *Args
	handle framework.FrameworkHandle
}

func (s *seal) Name() string {
	return Name
}

func New(configuration *runtime.Unknown, f framework.FrameworkHandle) (framework.Plugin, error) {
	args := &Args{}
	if err := framework.DecodeInto(configuration, args); err != nil {
		return nil, err
	}
	klog.V(3).Infof("get plugin config args: %+v", args)
	return &seal{
		args:   args,
		handle: f,
	}, nil
}

func (s *seal) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	//files, _ := rconfig.OpenJson("/node_score_list.json")
	nodeScore, _ := score.Score(nodeName)
	//nodeScore := files.Get(nodeName)
	return nodeScore, framework.NewStatus(framework.Success, "")
}

func (s *seal) NormalizeScore(ctx context.Context, state *framework.CycleState, p *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	var (
		highest int64 = 0
		lowest = scores[0].Score
	)
	for _, nodeScore := range scores {
		if nodeScore.Score < lowest {
			lowest = nodeScore.Score
		}
	}
	if lowest < 0 {
		for i := range scores {
			scores[i].Score -= lowest
		}
	}
	for _, nodeScore := range scores {
		if nodeScore.Score > highest {
			highest = nodeScore.Score
		}
	}
	// Set Range to [0-100]
	for i, nodeScore := range scores {
		scores[i].Score = nodeScore.Score * framework.MaxNodeScore / highest
	}
	return framework.NewStatus(framework.Success, "")
}

func (s *seal) ScoreExtensions() framework.ScoreExtensions {
	return s
}
