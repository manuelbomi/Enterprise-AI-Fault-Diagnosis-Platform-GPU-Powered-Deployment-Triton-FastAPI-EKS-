{{- define "enterprise-deploy.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "enterprise-deploy.fullname" -}}
{{- printf "%s-%s" (include "enterprise-deploy.name" .) .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
