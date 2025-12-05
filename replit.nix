{ pkgs }: {
  deps = [
    pkgs.python39
    pkgs.postgresql
    pkgs.python39Packages.pip
  ];
} 