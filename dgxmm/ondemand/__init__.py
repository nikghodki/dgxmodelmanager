"""On-demand deployments: started by the first inference request, stopped after an idle timeout (controller.py)."""

# Sent by LiteLLM on its health checks of on-demand entries (model_info.health_check_params, dgxmm/litellm.py): the
# gateway answers them without waking the model and without counting them as activity.
HEALTH_HEADER = "x-dgxmm-health-check"
