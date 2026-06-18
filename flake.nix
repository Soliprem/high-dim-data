{
  description = "Simple Devshell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system} = {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python3.withPackages
            (python-pkgs: with python-pkgs; [
              matplotlib
              numpy
              pandas
              scikit-learn
              scipy
              seaborn
              tqdm
            ]))
        ];
      };
    };
  };
}
