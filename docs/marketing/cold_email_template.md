Subject: Project Aurelius — physics-grounded EA for electrolyte discovery, Spearman ρ=0.76 LUMO validation

Hi [Name],

I'm reaching out to share Project Aurelius, an open-source evolutionary algorithm pipeline we've built for battery electrolyte discovery that combines a hybrid quantum (xTB/TOM) and fragment-additivity oracle to screen molecules without ML frameworks. Our external validation against published experimental data shows Spearman ρ = 0.76 for LUMO predictions and positive rank correlation across all five benchmarked properties (dielectric, viscosity, donor number, HOMO, LUMO). The pipeline is CLI-first, self-verifying (a repository-level objective function penalizes code complexity to ensure maintainability), and available under MIT at github.com/tomwolfe/ProjectAurelius — I'd welcome your team's perspective on how this could integrate with your computational workflow.
