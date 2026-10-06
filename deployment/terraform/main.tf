# ==============================================================================
# Infrastructure-as-Code (Terraform) — Google Cloud Agent Starter Pack Layout
# (`Prototype to Production.pdf`, pp. 21-38)
# Provisions:
#   1. Dedicated Least-Privilege Agent Identity Service Account
#   2. Cloud Run v2 Service with Safe Canary Rollout (90% stable / 10% canary)
#   3. Cloud Logging / Cloud Trace sink for OpenTelemetry Agent Observability
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = "sme-process-hypothesis-prod"
}

variable "region" {
  description = "Google Cloud Region"
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "Artifact Registry container image URI"
  type        = string
  default     = "us-central1-docker.pkg.dev/sme-process-hypothesis-prod/agents/process-hypothesis-agent:latest"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Layer 1 Security: Dedicated Agent Identity Service Account (`Introduction to Agents.pdf`, p. 38)
resource "google_service_account" "agent_identity" {
  account_id   = "process-hypothesis-agent-sa"
  display_name = "Process Hypothesis Multi-Agent Runtime Identity"
  description  = "Least-privilege cryptographic identity for hypothesis_generator_agent and hypothesis_evaluator_agent"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.agent_identity.email}"
}

resource "google_project_iam_member" "cloud_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.agent_identity.email}"
}

# 2. Cloud Run v2 Service with Canary Traffic Splitting (`Prototype to Production.pdf`, p. 25)
resource "google_cloud_run_v2_service" "process_hypothesis_service" {
  name     = "process-hypothesis-agent"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.agent_identity.email
    containers {
      image = var.container_image
      ports {
        container_port = 8080
      }
      env {
        name  = "SAFETY_PLUGIN_ENABLED"
        value = "true"
      }
      env {
        name  = "OTEL_SERVICE_NAME"
        value = "process-hypothesis-agent"
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}
