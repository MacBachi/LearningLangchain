# ./flake.nix
{
  description = "Learning LangChain";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        shell = pkgs.zsh;
        buildInputs = with pkgs; [
          gcc
          git
          poetry
          pkg-config
          pre-commit
          python313
        ];

        shellHook = ''
          set -e
          echo "✅ Python-Entwicklungsumgebung (Python ${pkgs.python313.version}) ist aktiv."

          poetry config virtualenvs.in-project true

          # Installation und Synchronisation der Abhängigkeiten
          if [ -f "pyproject.toml" ]; then
            echo "📦 Installiere/Synchronisiere Abhängigkeiten mit Poetry..."
            poetry install --no-root
                      
            # Aktivierung des Poetry-spezifischen virtuellen Environments
            source $(poetry env info --path)/bin/activate
          else
            echo "⚠️ pyproject.toml nicht gefunden. Willst du poetry für die Paketverwaltung nutzen, starte einmalig 'poetry init' und verwende dann Poetry für die Paketverwaltung."
          fi

          # Automatisches Erstellen und Aktivieren des venv
          if [ ! -d ".venv" ]; then
            echo "🐍 Erstelle virtuelles Environment im Ordner .venv..."
            python -m venv .venv
          fi
          source .venv/bin/activate

          # Poetry venv im Projektordner erzwingen (optional)
          poetry config virtualenvs.in-project true

          export MY_API_KEY="dein_schlüssel_hier"

          if [ -f ".pre-commit-config.yaml" ]; then
            pre-commit install
          fi
          
          if [ -f ".env" ]; then
            source .env
          fi
          
          if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
          fi
          set +e
        '';
      };
    };
}
