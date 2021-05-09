package register

import (
	"github.com/xzyu0106/Seal-Scheduler/pkg/seal"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
)

func Register() *cobra.Command {
	return app.NewSchedulerCommand(
		app.WithPlugin(seal.Name, seal.New),
	)
}
